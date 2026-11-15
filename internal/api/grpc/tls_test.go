package grpc

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/zafrem/bastion-navigator/internal/models"
)

// ─── test PKI ────────────────────────────────────────────────────────────────

type pki struct {
	dir                           string
	caKey                         *ecdsa.PrivateKey
	caCert                        *x509.Certificate
	caFile, certFile, keyFile     string
	clientCertFile, clientKeyFile string
	serial                        int64
}

func writePEM(t *testing.T, path, typ string, der []byte) {
	t.Helper()
	if err := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: typ, Bytes: der}), 0o600); err != nil {
		t.Fatal(err)
	}
}

func newCA(t *testing.T, name string) (*ecdsa.PrivateKey, *x509.Certificate, []byte) {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: name},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour),
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, _ := x509.ParseCertificate(der)
	return key, cert, der
}

func (p *pki) issue(t *testing.T, certPath, keyPath, cn string, server bool) {
	t.Helper()
	p.serial++
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(100 + p.serial), Subject: pkix.Name{CommonName: cn},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature,
	}
	if server {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
		tmpl.DNSNames = []string{"localhost"}
		tmpl.IPAddresses = []net.IP{net.ParseIP("127.0.0.1")}
	} else {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, p.caCert, &key.PublicKey, p.caKey)
	if err != nil {
		t.Fatal(err)
	}
	kder, _ := x509.MarshalECPrivateKey(key)
	writePEM(t, certPath, "CERTIFICATE", der)
	writePEM(t, keyPath, "EC PRIVATE KEY", kder)
}

func newPKI(t *testing.T) *pki {
	t.Helper()
	p := &pki{dir: t.TempDir()}
	var der []byte
	p.caKey, p.caCert, der = newCA(t, "test-ca")
	p.caFile = filepath.Join(p.dir, "ca.pem")
	writePEM(t, p.caFile, "CERTIFICATE", der)
	p.certFile, p.keyFile = filepath.Join(p.dir, "server.pem"), filepath.Join(p.dir, "server.key")
	p.clientCertFile, p.clientKeyFile = filepath.Join(p.dir, "client.pem"), filepath.Join(p.dir, "client.key")
	p.issue(t, p.certFile, p.keyFile, "server-1", true)
	p.issue(t, p.clientCertFile, p.clientKeyFile, "client-1", false)
	return p
}

// rotate replaces the server key pair on disk with a new one from the same CA.
func (p *pki) rotate(t *testing.T) {
	t.Helper()
	p.issue(t, p.certFile, p.keyFile, "server-2", true)
	later := time.Now().Add(5 * time.Second) // make sure the mtime differs on coarse filesystems
	_ = os.Chtimes(p.certFile, later, later)
	_ = os.Chtimes(p.keyFile, later, later)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

func serveOn(t *testing.T, s *Server) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = s.Serve(lis) }()
	t.Cleanup(s.grpcServer.Stop)
	return lis.Addr().String()
}

func tlsClientConfig(t *testing.T, caFile, certFile, keyFile string) *tls.Config {
	t.Helper()
	pemBytes, _ := os.ReadFile(caFile)
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(pemBytes)
	cfg := &tls.Config{RootCAs: pool, ServerName: "localhost", MinVersion: tls.VersionTLS12, NextProtos: []string{"h2"}}
	if certFile != "" {
		pair, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			t.Fatal(err)
		}
		cfg.Certificates = []tls.Certificate{pair}
	}
	return cfg
}

// handshake dials with TLS. TLS 1.3 reports a rejected client certificate only on the first read,
// so it reads briefly: a timeout means the server accepted the connection.
func handshake(addr string, cfg *tls.Config) error {
	conn, err := tls.DialWithDialer(&net.Dialer{Timeout: 3 * time.Second}, "tcp", addr, cfg)
	if err != nil {
		return err
	}
	defer conn.Close()
	_ = conn.SetReadDeadline(time.Now().Add(400 * time.Millisecond))
	if _, err := conn.Read(make([]byte, 1)); err != nil {
		var ne net.Error
		if errors.As(err, &ne) && ne.Timeout() {
			return nil
		}
		return err
	}
	return nil
}

// speaksPlaintextHTTP2 sends an HTTP/2 preface and an empty SETTINGS frame without TLS and reports
// whether the server answers with a SETTINGS frame, i.e. whether it accepts plaintext gRPC.
func speaksPlaintextHTTP2(addr string) bool {
	c, err := net.DialTimeout("tcp", addr, 3*time.Second)
	if err != nil {
		return false
	}
	defer c.Close()
	_, _ = c.Write([]byte("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"))
	_, _ = c.Write([]byte{0, 0, 0, 4, 0, 0, 0, 0, 0})
	_ = c.SetReadDeadline(time.Now().Add(time.Second))
	buf := make([]byte, 9)
	if _, err := io.ReadFull(c, buf); err != nil {
		return false
	}
	return buf[3] == 0x04
}

func peerSerial(t *testing.T, addr string) int64 {
	t.Helper()
	conn, err := tls.Dial("tcp", addr, &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"h2"}}) //nolint:gosec // reading the presented certificate only
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	return conn.ConnectionState().PeerCertificates[0].SerialNumber.Int64()
}

func newTestServer(t *testing.T) *Server {
	t.Helper()
	return New(nil, 0)
}

// ─── tests ───────────────────────────────────────────────────────────────────

func TestTLS_NotConfiguredMeansNoTLS(t *testing.T) {
	s := newTestServer(t)
	if err := s.EnableTLS("", "", ""); err != nil {
		t.Fatalf("EnableTLS with nothing configured: %v", err)
	}
	if s.TLSEnabled() {
		t.Error("TLSEnabled is true although nothing was configured")
	}
	if !speaksPlaintextHTTP2(serveOn(t, s)) {
		t.Error("an unconfigured server should accept plaintext gRPC (the probe itself is broken otherwise)")
	}
}

func TestTLS_MisconfigurationIsAnError(t *testing.T) {
	p := newPKI(t)
	garbage := filepath.Join(p.dir, "garbage.pem")
	_ = os.WriteFile(garbage, []byte("not a pem"), 0o600)
	cases := map[string][3]string{
		"cert without key":           {p.certFile, "", ""},
		"key without cert":           {"", p.keyFile, ""},
		"client CA without a server": {"", "", p.caFile},
		"missing cert file":          {filepath.Join(p.dir, "nope.pem"), p.keyFile, ""},
		"mismatched key pair":        {p.certFile, p.clientKeyFile, ""},
		"garbage client CA":          {p.certFile, p.keyFile, garbage},
	}
	for name, c := range cases {
		if err := newTestServer(t).EnableTLS(c[0], c[1], c[2]); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}

func TestTLS_ServesOverTLSAndRejectsPlaintext(t *testing.T) {
	p := newPKI(t)
	s := newTestServer(t)
	if err := s.EnableTLS(p.certFile, p.keyFile, ""); err != nil {
		t.Fatal(err)
	}
	if !s.TLSEnabled() {
		t.Fatal("TLSEnabled should be true")
	}
	addr := serveOn(t, s)

	if err := handshake(addr, tlsClientConfig(t, p.caFile, "", "")); err != nil {
		t.Errorf("TLS client trusting the CA: %v", err)
	}
	if speaksPlaintextHTTP2(addr) {
		t.Error("a TLS server must not answer plaintext HTTP/2")
	}
	_, _, otherDER := newCA(t, "other-ca")
	other := filepath.Join(p.dir, "other-ca.pem")
	writePEM(t, other, "CERTIFICATE", otherDER)
	if err := handshake(addr, tlsClientConfig(t, other, "", "")); err == nil {
		t.Error("a client that does not trust the server certificate must fail")
	}
}

func TestTLS_MutualTLSRequiresAClientCertificate(t *testing.T) {
	p := newPKI(t)
	s := newTestServer(t)
	if err := s.EnableTLS(p.certFile, p.keyFile, p.caFile); err != nil {
		t.Fatal(err)
	}
	addr := serveOn(t, s)

	if err := handshake(addr, tlsClientConfig(t, p.caFile, "", "")); err == nil {
		t.Error("mTLS: a client without a certificate must be rejected")
	}
	if err := handshake(addr, tlsClientConfig(t, p.caFile, p.clientCertFile, p.clientKeyFile)); err != nil {
		t.Errorf("mTLS: client with a certificate from the trusted CA: %v", err)
	}
	rogue := &pki{dir: t.TempDir()}
	rogue.caKey, rogue.caCert, _ = newCA(t, "rogue-ca")
	rc, rk := filepath.Join(rogue.dir, "c.pem"), filepath.Join(rogue.dir, "c.key")
	rogue.issue(t, rc, rk, "rogue", false)
	if err := handshake(addr, tlsClientConfig(t, p.caFile, rc, rk)); err == nil {
		t.Error("mTLS: a client certificate from an untrusted CA must be rejected")
	}
}

func TestTLS_RotatedCertificateIsPickedUpWithoutARestart(t *testing.T) {
	p := newPKI(t)
	s := newTestServer(t)
	if err := s.EnableTLS(p.certFile, p.keyFile, ""); err != nil {
		t.Fatal(err)
	}
	addr := serveOn(t, s)

	before := peerSerial(t, addr)
	p.rotate(t)
	if after := peerSerial(t, addr); after == before {
		t.Fatalf("server still presents the old certificate (serial %d) after the files were replaced", after)
	}
	if err := handshake(addr, tlsClientConfig(t, p.caFile, "", "")); err != nil {
		t.Errorf("handshake with the rotated certificate: %v", err)
	}
}

func TestTLS_BrokenReplacementKeepsTheWorkingCertificate(t *testing.T) {
	p := newPKI(t)
	s := newTestServer(t)
	if err := s.EnableTLS(p.certFile, p.keyFile, ""); err != nil {
		t.Fatal(err)
	}
	addr := serveOn(t, s)
	before := peerSerial(t, addr)

	_ = os.WriteFile(p.certFile, []byte("half-written"), 0o600)
	later := time.Now().Add(5 * time.Second)
	_ = os.Chtimes(p.certFile, later, later)
	if got := peerSerial(t, addr); got != before {
		t.Errorf("a broken replacement changed the served certificate (%d -> %d)", before, got)
	}
}

func TestGRPCAuth_RequiresBearerTokenWhenConfigured(t *testing.T) {
	s := newTestServer(t)
	s.EnableAuth("nav-grpc-secret")
	if !s.AuthEnabled() {
		t.Fatal("AuthEnabled should be true")
	}
	addr := serveOn(t, s)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(jsonCodec{})))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	// Health succeeds without token.
	var healthOut models.HealthStatus
	if err := conn.Invoke(context.Background(), "/bastion.navigator.v1.NavigatorService/Health", &models.HealthRequest{}, &healthOut); err != nil {
		t.Fatalf("Health without token failed: %v", err)
	}

	// Protected RPC fails without token.
	var colOut models.CollectionsResponse
	err = conn.Invoke(context.Background(), "/bastion.navigator.v1.NavigatorService/GetCollections", &models.CollectionsRequest{}, &colOut)
	if status.Code(err) != codes.Unauthenticated {
		t.Errorf("GetCollections without token: got %v, want Unauthenticated", status.Code(err))
	}

	// Protected RPC fails with wrong token.
	badCtx := metadata.AppendToOutgoingContext(context.Background(), "authorization", "Bearer wrong")
	err = conn.Invoke(badCtx, "/bastion.navigator.v1.NavigatorService/GetCollections", &models.CollectionsRequest{}, &colOut)
	if status.Code(err) != codes.Unauthenticated {
		t.Errorf("GetCollections with wrong token: got %v, want Unauthenticated", status.Code(err))
	}
}


func TestGRPCAuth_ReflectionRequiresToken(t *testing.T) {
	s := newTestServer(t)
	s.EnableAuth("nav-grpc-secret")
	addr := serveOn(t, s)
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(jsonCodec{})))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	stream, err := conn.NewStream(ctx, &grpc.StreamDesc{StreamName: "ServerReflectionInfo", ServerStreams: true, ClientStreams: true},
		"/grpc.reflection.v1.ServerReflection/ServerReflectionInfo")
	if err == nil {
		var out struct{}
		err = stream.RecvMsg(&out)
	}
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("reflection without token: got %v, want Unauthenticated", err)
	}
}

// The JSON codec is scoped to this server; it must not replace the process-wide protobuf codec.
func TestJSONCodecDoesNotReplaceProtoCodec(t *testing.T) {
	if c := encoding.GetCodec("proto"); c != nil {
		if _, isJSON := c.(jsonCodec); isJSON {
			t.Fatal(`jsonCodec is registered as "proto", replacing the global protobuf codec`)
		}
	}
}
