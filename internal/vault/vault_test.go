package vault

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zafrem/bastion-navigator/internal/cache"
	"github.com/zafrem/bastion-navigator/internal/models"
)

// alice is a fully identified user: Vault needs department and role to answer for her.
var alice = models.UserContext{UserID: "alice", TenantID: "t1", Department: "sales", Roles: []string{"analyst"}}

func userWith(id string) models.UserContext {
	u := alice
	u.UserID = id
	return u
}

// fakeVault answers /v1/vault/permissions/{user} and records what it was sent.
type fakeVault struct {
	srv        *httptest.Server
	gotAuth    []string
	gotPaths   []string
	gotQueries []url.Values
	status     int
	body       string
}

func newFakeVault(t *testing.T, status int, body string) *fakeVault {
	t.Helper()
	f := &fakeVault{status: status, body: body}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.gotAuth = append(f.gotAuth, r.Header.Get("Authorization"))
		f.gotPaths = append(f.gotPaths, r.URL.EscapedPath())
		f.gotQueries = append(f.gotQueries, r.URL.Query())
		w.WriteHeader(f.status)
		_, _ = w.Write([]byte(f.body))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func TestHTTP_SendsBearerTokenToThePermissionsEndpoint(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01","DC-03"]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithToken("svc-token"))
	got, err := c.AllowedCategories(context.Background(), alice)
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
	if _, err := c.AllowedCategories(context.Background(), userWith("a/../admin")); err != nil {
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
	_, _ = c.AllowedCategories(context.Background(), alice)
	_ = os.WriteFile(file, []byte("second\n"), 0o600)
	_, _ = c.AllowedCategories(context.Background(), alice)
	if !reflect.DeepEqual(f.gotAuth, []string{"Bearer first", "Bearer second"}) {
		t.Errorf("Authorization headers = %v", f.gotAuth)
	}
}

func TestHTTP_UnreadableTokenFileFailsWithoutCallingVault(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":[]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithTokenFile(filepath.Join(t.TempDir(), "missing")))
	if _, err := c.AllowedCategories(context.Background(), alice); err == nil {
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
		if cats, err := c.AllowedCategories(context.Background(), alice); err == nil {
			t.Errorf("%s: want an error, got categories %v", tc.name, cats)
		}
	}
}

func TestHTTP_AsksVaultAboutTheUsersDepartmentAndRoleNotTheServices(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01"]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute, WithToken("svc"))
	if _, err := c.AllowedCategories(context.Background(), alice); err != nil {
		t.Fatal(err)
	}
	q := f.gotQueries[0]
	if q.Get("department") != "sales" || q.Get("role") != "analyst" || q.Get("tenant_id") != "t1" {
		t.Errorf("query = %v, want department, role and tenant of the user", q)
	}
}

func TestHTTP_MissingIdentityDeniesWithoutCallingVault(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01","DC-02","DC-03"]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute)
	for name, u := range map[string]models.UserContext{
		"no user":       {Department: "sales", Roles: []string{"analyst"}},
		"no department": {UserID: "alice", Roles: []string{"analyst"}},
		"no roles":      {UserID: "alice", Department: "sales"},
		"blank roles":   {UserID: "alice", Department: "sales", Roles: []string{""}},
	} {
		cats, err := c.AllowedCategories(context.Background(), u)
		if err != nil || cats == nil || len(cats) != 0 {
			t.Errorf("%s: got %v, %v; want a non-nil empty set (no access)", name, cats, err)
		}
	}
	if len(f.gotPaths) != 0 {
		t.Errorf("Vault was called %d times for unidentifiable users", len(f.gotPaths))
	}
}

func TestHTTP_UnionsCategoriesAcrossRoles(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		if r.URL.Query().Get("role") == "analyst" {
			_, _ = w.Write([]byte(`{"allowed_categories":["DC-01"]}`))
			return
		}
		_, _ = w.Write([]byte(`{"allowed_categories":["DC-01","DC-03"]}`))
	}))
	defer srv.Close()
	c := NewHTTP(srv.URL, nil, time.Minute)
	u := alice
	u.Roles = []string{"analyst", "hr_admin"}
	got, err := c.AllowedCategories(context.Background(), u)
	if err != nil || !reflect.DeepEqual(got, []string{"customer_data", "hr_data"}) {
		t.Errorf("got %v, %v; want the deduplicated union", got, err)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("Vault calls = %d, want one per role", calls)
	}
}

func TestHTTP_EmptyAnswerIsNoAccessNotUnrestricted(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":[]}`)
	c := NewHTTP(f.srv.URL, nil, time.Minute)
	got, err := c.AllowedCategories(context.Background(), alice)
	if err != nil || got == nil || len(got) != 0 {
		t.Fatalf("got %v, %v; want a non-nil empty set", got, err)
	}
	in := []models.SearchResult{{DocumentID: "a", Category: "customer_data"}}
	if out, removed := c.FilterResults(in, got); len(out) != 0 || removed != 1 {
		t.Errorf("an empty allowed set must remove everything, got %v (removed %d)", out, removed)
	}
}

func TestHTTP_PermissionsAreCachedOnlyUntilTheTTLExpires(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01"]}`)
	now := time.Now()
	c := NewHTTP(f.srv.URL, cache.NewInMemory(), time.Minute).(*httpClient)
	c.now = func() time.Time { return now }
	for i := 0; i < 3; i++ {
		if _, err := c.AllowedCategories(context.Background(), alice); err != nil {
			t.Fatal(err)
		}
	}
	if len(f.gotPaths) != 1 {
		t.Fatalf("Vault calls within the TTL = %d, want 1", len(f.gotPaths))
	}
	now = now.Add(2 * time.Minute) // a revocation in Vault must take effect once the entry has expired
	if _, err := c.AllowedCategories(context.Background(), alice); err != nil {
		t.Fatal(err)
	}
	if len(f.gotPaths) != 2 {
		t.Errorf("Vault calls after expiry = %d, want 2", len(f.gotPaths))
	}
}

func TestHTTP_CacheIsKeyedByDepartmentAndRoles(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01"]}`)
	c := NewHTTP(f.srv.URL, cache.NewInMemory(), time.Minute)
	other := alice
	other.Department = "hr"
	_, _ = c.AllowedCategories(context.Background(), alice)
	_, _ = c.AllowedCategories(context.Background(), other)
	if len(f.gotPaths) != 2 {
		t.Errorf("a different department reused another's cached answer (%d Vault calls)", len(f.gotPaths))
	}
}

func TestHTTP_NoCachingWithoutATTL(t *testing.T) {
	f := newFakeVault(t, 200, `{"allowed_categories":["DC-01"]}`)
	c := NewHTTP(f.srv.URL, cache.NewInMemory(), 0)
	_, _ = c.AllowedCategories(context.Background(), alice)
	_, _ = c.AllowedCategories(context.Background(), alice)
	if len(f.gotPaths) != 2 {
		t.Errorf("Vault calls = %d, want 2 when the TTL is zero", len(f.gotPaths))
	}
}

func TestMockIsUnrestricted(t *testing.T) {
	cats, err := NewMock().AllowedCategories(context.Background(), alice)
	if err != nil || cats != nil {
		t.Errorf("mock = %v, %v; want nil (unrestricted)", cats, err)
	}
	in := []models.SearchResult{{DocumentID: "a", Category: "x"}}
	if out, removed := NewMock().FilterResults(in, nil); len(out) != 1 || removed != 0 {
		t.Errorf("nil categories must keep everything, got %v (removed %d)", out, removed)
	}
}
