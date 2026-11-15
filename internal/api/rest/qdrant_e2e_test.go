package rest

// End-to-end test: REST API -> orchestrator -> real Qdrant, with the real Vault HTTP client talking to a
// fake Vault. It needs a running Qdrant and is skipped unless NAVIGATOR_E2E_QDRANT is set:
//
//	podman run -d --rm -p 16333:6333 docker.io/qdrant/qdrant:v1.12.4
//	NAVIGATOR_E2E_QDRANT=localhost:16333 go test ./internal/api/rest -run QdrantE2E -v
//
// CI runs it against a Qdrant service container.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/zafrem/bastion-navigator/internal/config"
	"github.com/zafrem/bastion-navigator/internal/models"
	"github.com/zafrem/bastion-navigator/internal/orchestrator"
	"github.com/zafrem/bastion-navigator/internal/searcher"
	"github.com/zafrem/bastion-navigator/internal/vault"
)

const e2eToken = "e2e-navigator-token"

// Fixed UUIDs, so results can be asserted by identity.
const (
	idAlpha   = "00000000-0000-0000-0000-00000000000a" // t1 customer, exact match
	idBravo   = "00000000-0000-0000-0000-00000000000b" // t1 customer, close match
	idCharlie = "00000000-0000-0000-0000-00000000000c" // t1 hr, close match
	idDelta   = "00000000-0000-0000-0000-00000000000d" // t2 customer, exact match (other tenant)
	idEcho    = "00000000-0000-0000-0000-00000000000e" // t1 manufacturing, poor match
)

type e2eEmbedder struct{}

func (e2eEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0, 0, 0}, nil
}
func (e2eEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{1, 0, 0, 0}
	}
	return out, nil
}

func qdrantDo(t *testing.T, host, method, path string, body any) {
	t.Helper()
	var rdr *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rdr = bytes.NewReader(b)
	} else {
		rdr = bytes.NewReader(nil)
	}
	req, _ := http.NewRequest(method, "http://"+host+path, rdr)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("qdrant %s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		t.Fatalf("qdrant %s %s: HTTP %d", method, path, resp.StatusCode)
	}
}

// seedQdrant creates a uniquely named collection with five points and removes it afterwards.
func seedQdrant(t *testing.T, host string) string {
	t.Helper()
	col := fmt.Sprintf("nav_e2e_%d", time.Now().UnixNano())
	qdrantDo(t, host, http.MethodPut, "/collections/"+col, map[string]any{
		"vectors": map[string]any{"size": 4, "distance": "Cosine"},
	})
	t.Cleanup(func() { qdrantDo(t, host, http.MethodDelete, "/collections/"+col, nil) })

	pt := func(id string, vec []float32, tenant, category, content string) map[string]any {
		return map[string]any{"id": id, "vector": vec, "payload": map[string]string{
			"tenant_id": tenant, "category": category, "content": content,
		}}
	}
	qdrantDo(t, host, http.MethodPut, "/collections/"+col+"/points?wait=true", map[string]any{
		"points": []map[string]any{
			pt(idAlpha, []float32{1, 0, 0, 0}, "t1", "customer_data", "alpha customer record"),
			pt(idBravo, []float32{0.9, 0.1, 0, 0}, "t1", "customer_data", "bravo customer record"),
			pt(idCharlie, []float32{0.95, 0.05, 0, 0}, "t1", "hr_data", "charlie hr record"),
			pt(idDelta, []float32{1, 0, 0, 0}, "t2", "customer_data", "delta customer record of another tenant"),
			pt(idEcho, []float32{0, 1, 0, 0}, "t1", "manufacturing_data", "echo manufacturing record"),
		},
	})
	return col
}

// fakeVault answers the permissions endpoint by department, like Vault's RBAC matrix would.
func fakeVault(t *testing.T) *httptest.Server {
	t.Helper()
	byDept := map[string][]string{
		"marketing":     {"DC-01"},
		"hr":            {"DC-03"},
		"manufacturing": {"DC-02"},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cats, ok := byDept[r.URL.Query().Get("department")]
		if !ok {
			cats = []string{}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"allowed_categories": cats})
	}))
	t.Cleanup(srv.Close)
	return srv
}

type e2eEnv struct{ srv *Server }

func newE2E(t *testing.T) *e2eEnv {
	t.Helper()
	host := os.Getenv("NAVIGATOR_E2E_QDRANT")
	if host == "" {
		t.Skip("NAVIGATOR_E2E_QDRANT not set: skipping the Qdrant end-to-end test")
	}
	seedQdrant(t, host)
	v := fakeVault(t)

	cfg := config.Defaults()
	cfg.SearchDefaults.MinScore = 0
	cfg.SearchDefaults.UseHybrid = false
	cfg.SearchDefaults.UseReranking = false
	orch := orchestrator.New(cfg, e2eEmbedder{}, searcher.NewQdrant([]string{host}), noopReranker{},
		vault.NewHTTP(v.URL, nil, 0))
	return &e2eEnv{srv: New(orch, 0, WithAuthToken(e2eToken))}
}

func (e *e2eEnv) search(t *testing.T, body map[string]any) (int, models.SearchResponse) {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/navigator/search", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e2eToken)
	w := httptest.NewRecorder()
	e.srv.routes().ServeHTTP(w, req)
	var resp models.SearchResponse
	if w.Code == http.StatusOK {
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode %q: %v", w.Body.String(), err)
		}
	}
	return w.Code, resp
}

func docIDs(r models.SearchResponse) []string {
	out := make([]string, 0, len(r.Results))
	for _, x := range r.Results {
		out = append(out, x.DocumentID)
	}
	return out
}

func sameSet(a, b []string) bool {
	a, b = append([]string(nil), a...), append([]string(nil), b...)
	sort.Strings(a)
	sort.Strings(b)
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func user(tenant, id, dept string) map[string]any {
	return map[string]any{"query": "customer record", "tenant_id": tenant, "user_id": id,
		"department": dept, "roles": []string{"manager"}}
}

func with(m map[string]any, k string, v any) map[string]any {
	out := map[string]any{}
	for kk, vv := range m {
		out[kk] = vv
	}
	out[k] = v
	return out
}

func TestQdrantE2E_RequiresTheBearerToken(t *testing.T) {
	e := newE2E(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/navigator/search", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	e.srv.routes().ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 without a token", w.Code)
	}
}

func TestQdrantE2E_TenantAndPermissionFiltering(t *testing.T) {
	e := newE2E(t)

	t.Run("marketing in t1 sees only t1 customer data, best match first", func(t *testing.T) {
		code, resp := e.search(t, user("t1", "alice", "marketing"))
		if code != 200 {
			t.Fatalf("status %d", code)
		}
		got := docIDs(resp)
		if len(got) != 2 || got[0] != idAlpha || got[1] != idBravo {
			t.Errorf("results = %v, want [alpha bravo]: tenant t2's delta, hr's charlie and manufacturing's echo must not appear", got)
		}
		if resp.Results[0].Score < resp.Results[1].Score {
			t.Errorf("results not ordered by score: %v then %v", resp.Results[0].Score, resp.Results[1].Score)
		}
		if resp.Results[0].Content != "alpha customer record" || resp.Results[0].Category != "customer_data" {
			t.Errorf("payload not carried through: %+v", resp.Results[0])
		}
		if resp.Metadata.FilteredOut != 2 { // charlie (hr) and echo (manufacturing) were retrieved, then removed by Vault
			t.Errorf("FilteredOut = %d, want 2", resp.Metadata.FilteredOut)
		}
	})

	t.Run("hr user sees only hr data", func(t *testing.T) {
		_, resp := e.search(t, user("t1", "carol", "hr"))
		if got := docIDs(resp); !sameSet(got, []string{idCharlie}) {
			t.Errorf("results = %v, want [charlie]", got)
		}
	})

	t.Run("the same marketing user in t2 sees only t2's document", func(t *testing.T) {
		_, resp := e.search(t, user("t2", "alice", "marketing"))
		if got := docIDs(resp); !sameSet(got, []string{idDelta}) {
			t.Errorf("results = %v, want [delta]: another tenant's documents must never leak", got)
		}
	})

	t.Run("an unknown department has no access and nothing is searched", func(t *testing.T) {
		code, resp := e.search(t, user("t1", "mallory", "nowhere"))
		if code != 200 || len(resp.Results) != 0 || resp.Metadata.Strategy != "denied" {
			t.Errorf("got %d %v (%s), want 200 with no results, strategy denied", code, docIDs(resp), resp.Metadata.Strategy)
		}
	})

	t.Run("a request without department and role is denied even though the user id is known", func(t *testing.T) {
		_, resp := e.search(t, map[string]any{"query": "customer record", "tenant_id": "t1", "user_id": "alice"})
		if len(resp.Results) != 0 {
			t.Errorf("results = %v, want none", docIDs(resp))
		}
	})

	t.Run("caller categories can narrow but not widen", func(t *testing.T) {
		_, narrowed := e.search(t, with(user("t1", "alice", "marketing"), "allowed_categories", []string{"customer_data", "hr_data"}))
		if got := docIDs(narrowed); len(got) != 2 {
			t.Errorf("results = %v, want alpha and bravo only (hr_data is not allowed for marketing)", got)
		}
		_, none := e.search(t, with(user("t1", "alice", "marketing"), "allowed_categories", []string{"hr_data"}))
		if len(none.Results) != 0 {
			t.Errorf("results = %v, want none: the caller asked only for a category Vault denies", docIDs(none))
		}
	})
}

func TestQdrantE2E_ScoreThresholdAndTopK(t *testing.T) {
	e := newE2E(t)

	_, strict := e.search(t, with(user("t1", "alice", "marketing"), "options", map[string]any{"min_score": 0.999}))
	if got := docIDs(strict); !sameSet(got, []string{idAlpha}) {
		t.Errorf("min_score 0.999 returned %v, want only the exact match", got)
	}

	_, one := e.search(t, with(user("t1", "alice", "marketing"), "options", map[string]any{"top_k": 1}))
	if got := docIDs(one); len(got) != 1 || got[0] != idAlpha {
		t.Errorf("top_k 1 returned %v, want [alpha]", got)
	}
}

func TestQdrantE2E_ListsCollections(t *testing.T) {
	e := newE2E(t)
	req := httptest.NewRequest(http.MethodGet, "/v1/navigator/collections", nil)
	req.Header.Set("Authorization", "Bearer "+e2eToken)
	w := httptest.NewRecorder()
	e.srv.routes().ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Collections []models.CollectionInfo `json:"collections"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range resp.Collections {
		if len(c.Name) > 8 && c.Name[:8] == "nav_e2e_" {
			found = true
			if c.VectorCount != 5 || c.Status == "" || c.Status == "unknown" {
				t.Errorf("collection info = %+v, want 5 vectors and a real status", c)
			}
		}
	}
	if !found {
		t.Errorf("seeded collection missing from %+v", resp.Collections)
	}
}
