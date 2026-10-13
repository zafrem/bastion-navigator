package vault

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// fakeVault answers /v1/vault/permissions/{user} and records what it was sent.
type fakeVault struct {
	srv      *httptest.Server
	gotAuth  []string
	gotPaths []string
	status   int
	body     string
}

func newFakeVault(t *testing.T, status int, body string) *fakeVault {
	t.Helper()
	f := &fakeVault{status: status, body: body}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.gotAuth = append(f.gotAuth, r.Header.Get("Authorization"))
		f.gotPaths = append(f.gotPaths, r.URL.EscapedPath())
		w.WriteHeader(f.status)
		_, _ = w.Write([]byte(f.body))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func TestHTTP_SendsBearerTokenToThePermissionsEndpoint(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01","DC-03"]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithToken("svc-token"))
	got, err := c.AllowedCategories(context.Background(), "alice")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"customer_data", "hr_data"}) {
		t.Errorf("categories = %v, want Vault IDs mapped to Navigator names", got)
	}
	if f.gotAuth[0] != "Bearer svc-token" {
		t.Errorf("Authorization = %q, want Bearer svc-token", f.gotAuth[0])
	}
	if f.gotPaths[0] != "/v1/vault/permissions/alice" {
		t.Errorf("path = %q, want /v1/vault/permissions/alice", f.gotPaths[0])
	}
}

func TestHTTP_EscapesTheUserID(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":[]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute)
	if _, err := c.AllowedCategories(context.Background(), "a/../admin"); err != nil {
		t.Fatal(err)
	}
	if f.gotPaths[0] != "/v1/vault/permissions/a%2F..%2Fadmin" {
		t.Errorf("path = %q, user ID was not escaped", f.gotPaths[0])
	}
}

func TestHTTP_TokenFileIsReReadOnEveryCall(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":[]}`)
	file := filepath.Join(t.TempDir(), "vault.token")
	_ = os.WriteFile(file, []byte("first\n"), 0o600)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithTokenFile(file))
	_, _ = c.AllowedCategories(context.Background(), "alice")
	_ = os.WriteFile(file, []byte("second\n"), 0o600)
	_, _ = c.AllowedCategories(context.Background(), "alice")
	if !reflect.DeepEqual(f.gotAuth, []string{"Bearer first", "Bearer second"}) {
		t.Errorf("Authorization headers = %v", f.gotAuth)
	}
}

func TestHTTP_UnreadableTokenFileFailsWithoutCallingVault(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":[]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithTokenFile(filepath.Join(t.TempDir(), "missing")))
	if _, err := c.AllowedCategories(context.Background(), "alice"); err == nil {
		t.Fatal("want an error when the configured token file cannot be read")
	}
	if len(f.gotAuth) != 0 {
		t.Error("Vault was called although the credentials could not be loaded")
	}
}

func TestHTTP_ErrorsAreReturnedNotSwallowed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{"unauthorized", 401, `{"error":"unauthorized"}`},
		{"server error", 500, `boom`},
		{"bad json", 200, `not json`},
		{"missing field", 200, `{"unexpected":true}`},
	} {
		f := newFakeVault(t, tc.status, tc.body)
		c := NewHTTP(f.srv.URL, nil, time.Minute)
		if cats, err := c.AllowedCategories(context.Background(), "alice"); err == nil {
			t.Errorf("%s: want an error, got categories %v", tc.name, cats)
		}
	}
}
