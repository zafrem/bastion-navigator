"""Navigator gRPC over TLS: server credentials, mutual TLS, rotation, and federation peers."""
from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import ssl
import time

import grpc
import pytest

from navigator.events import Publisher
from navigator.federation import PeerClient, PeerConfig
from navigator.grpc_server import GRPCServer
from navigator.hooks import HookManager
from navigator.models import SearchRequest
from navigator.tls import TLSConfigError, server_credentials
from tests._pki import PKI, needs_openssl
from tests.test_navigator_auth import _Orch

pytestmark = needs_openssl

HEALTH = "/bastion.navigator.v1.NavigatorService/Health"
SEARCH = "/bastion.navigator.v1.NavigatorService/Search"
TOKEN = "navigator-shared-token-0123456789"


# ─── helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def pki(tmp_path):
    return PKI(tmp_path)


@pytest.fixture()
def serve():
    servers = []

    def start(*, token="", cert="", key="", client_ca="") -> int:
        gs = GRPCServer(_Orch(), Publisher(""), HookManager(), auth_token=token,
                        tls_cert=str(cert) if cert else "", tls_key=str(key) if key else "",
                        tls_client_ca=str(client_ca) if client_ca else "")
        srv = gs._make_server()
        port = gs._add_port(srv, "localhost:0")
        srv.start()
        servers.append(srv)
        return port

    yield start
    for s in servers:
        s.stop(0)


def _read(p) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def _tls_channel(port, pki, *, ca=None, client=False, cert=None, key=None):
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read(ca or pki.ca_cert),
        private_key=_read(key or pki.client_key) if (client or key) else None,
        certificate_chain=_read(cert or pki.client_cert) if (client or cert) else None,
    )
    return grpc.secure_channel(f"localhost:{port}", creds)


def _call(channel, method=HEALTH, metadata=None):
    return channel.unary_unary(method)(b"{}", metadata=metadata or [], timeout=5)


def _served_cert_digest(port) -> str:
    """SHA-256 of the certificate the server presents (no verification)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock, ctx.wrap_socket(sock) as tls:
        return hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()


# ─── server credentials ───────────────────────────────────────────────────────

def test_no_tls_settings_means_plaintext(serve):
    assert server_credentials("", "", "") is None
    port = serve()
    assert _call(grpc.insecure_channel(f"localhost:{port}"))


def test_misconfiguration_raises(pki, tmp_path):
    garbage = tmp_path / "garbage.pem"
    garbage.write_text("not a pem")
    cases = {
        "cert without key": (str(pki.server_cert), "", ""),
        "key without cert": ("", str(pki.server_key), ""),
        "client CA without a server key pair": ("", "", str(pki.ca_cert)),
        "missing cert file": (str(tmp_path / "nope.pem"), str(pki.server_key), ""),
        "mismatched key pair": (str(pki.server_cert), str(pki.client_key), ""),
        "garbage client CA": (str(pki.server_cert), str(pki.server_key), str(garbage)),
    }
    for name, args in cases.items():
        with pytest.raises(TLSConfigError):
            server_credentials(*args)
        assert name  # (the case name is for readability of a failure)


def test_grpc_server_refuses_bad_tls_settings_at_construction(pki):
    with pytest.raises(TLSConfigError):
        GRPCServer(_Orch(), Publisher(""), HookManager(), tls_cert=str(pki.server_cert))


# ─── TLS and mutual TLS ───────────────────────────────────────────────────────

def test_serves_over_tls_and_rejects_plaintext(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    assert _call(_tls_channel(port, pki))
    with pytest.raises(grpc.RpcError) as e:
        _call(grpc.insecure_channel(f"localhost:{port}"))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


def test_client_that_does_not_trust_the_server_certificate_fails(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    other_ca, _ = pki.other_ca()
    with pytest.raises(grpc.RpcError) as e:
        _call(_tls_channel(port, pki, ca=other_ca))
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE


def test_mutual_tls_requires_a_trusted_client_certificate(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key, client_ca=pki.ca_cert)
    with pytest.raises(grpc.RpcError):
        _call(_tls_channel(port, pki))                                   # no client certificate
    assert _call(_tls_channel(port, pki, client=True))                   # certificate from the trusted CA
    rc, rk = pki.rogue_client()
    with pytest.raises(grpc.RpcError):
        _call(_tls_channel(port, pki, cert=rc, key=rk))                  # certificate from another CA


def test_bearer_token_still_applies_over_tls(pki, serve):
    port = serve(token=TOKEN, cert=pki.server_cert, key=pki.server_key)
    ch = _tls_channel(port, pki)
    with pytest.raises(grpc.RpcError) as e:
        _call(ch, SEARCH)
    assert e.value.code() == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.RpcError) as e:                              # past auth: the empty payload is rejected by the handler
        _call(ch, SEARCH, [("authorization", f"Bearer {TOKEN}")])
    assert e.value.code() != grpc.StatusCode.UNAUTHENTICATED


# ─── rotation ─────────────────────────────────────────────────────────────────

def test_rotated_certificate_is_picked_up_without_a_restart(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    before = _served_cert_digest(port)
    pki.rotate_server()
    after = _served_cert_digest(port)
    assert after != before, "the server still presents the old certificate after the files were replaced"
    assert _call(_tls_channel(port, pki))


def test_broken_replacement_keeps_the_working_certificate(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    before = _served_cert_digest(port)
    pki.server_cert.write_text("half-written")
    later = time.time() + 5
    os.utime(pki.server_cert, (later, later))
    assert _served_cert_digest(port) == before


# ─── federation: the peer client over TLS ─────────────────────────────────────

def _search(peer: PeerConfig):
    req = SearchRequest(request_id="r1", tenant_id="acme", query="q")
    return asyncio.run(PeerClient(peer, "node-a", 3000).search(req, 1))


def test_peer_client_reaches_a_tls_peer(pki, serve):
    port = serve(token=TOKEN, cert=pki.server_cert, key=pki.server_key)
    peer = PeerConfig(id="b", endpoint=f"localhost:{port}", auth_token=TOKEN, tls=True, tls_ca_file=str(pki.ca_cert))
    assert _search(peer) is not None


def test_peer_client_without_tls_cannot_reach_a_tls_peer(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    assert _search(PeerConfig(id="b", endpoint=f"localhost:{port}")) is None


def test_peer_client_rejects_a_peer_it_does_not_trust(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    other_ca, _ = pki.other_ca()
    peer = PeerConfig(id="b", endpoint=f"localhost:{port}", tls=True, tls_ca_file=str(other_ca))
    assert _search(peer) is None


def test_peer_client_presents_its_certificate_for_mutual_tls(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key, client_ca=pki.ca_cert)
    base = dict(id="b", endpoint=f"localhost:{port}", tls=True, tls_ca_file=str(pki.ca_cert))
    assert _search(PeerConfig(**base)) is None
    assert _search(PeerConfig(**base, tls_client_cert=str(pki.client_cert), tls_client_key=str(pki.client_key))) is not None


def test_peer_client_can_override_the_expected_server_name(pki, serve):
    port = serve(cert=pki.server_cert, key=pki.server_key)
    # Connect by IP but verify the certificate against its "localhost" name.
    peer = PeerConfig(id="b", endpoint=f"127.0.0.1:{port}", tls=True, tls_ca_file=str(pki.ca_cert),
                      tls_server_name="localhost")
    assert _search(peer) is not None
