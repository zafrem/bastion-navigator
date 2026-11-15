package orchestrator

import (
	"context"
	"errors"
	"testing"

	"github.com/zafrem/bastion-navigator/internal/config"
	"github.com/zafrem/bastion-navigator/internal/models"
	"github.com/zafrem/bastion-navigator/internal/vault"
)

type fakeEmbedder struct{ calls int }

func (f *fakeEmbedder) Embed(context.Context, string) ([]float32, error) {
	f.calls++
	return []float32{1}, nil
}
func (f *fakeEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) { return nil, nil }

type fakeSearcher struct{ results []models.SearchResult }

func (s *fakeSearcher) VectorSearch(context.Context, string, []float32, map[string]string, int, float64) ([]models.SearchResult, error) {
	return append([]models.SearchResult(nil), s.results...), nil
}
func (s *fakeSearcher) SparseSearch(context.Context, string, string, map[string]string, int) ([]models.SearchResult, error) {
	return nil, nil
}
func (s *fakeSearcher) Collections(context.Context) ([]models.CollectionInfo, error) {
	return []models.CollectionInfo{{Name: "docs"}}, nil
}

type fakeReranker struct{}

func (fakeReranker) Rerank(_ context.Context, _ string, c []models.SearchResult, _ int) ([]models.SearchResult, error) {
	return c, nil
}

// fakeVault answers AllowedCategories with fixed categories/error and filters like the real client.
type fakeVault struct {
	cats []string
	err  error
	got  models.UserContext
}

func (f *fakeVault) AllowedCategories(_ context.Context, u models.UserContext) ([]string, error) {
	f.got = u
	return f.cats, f.err
}
func (f *fakeVault) FilterResults(r []models.SearchResult, c []string) ([]models.SearchResult, int) {
	return vault.NewMock().FilterResults(r, c)
}

func docs() []models.SearchResult {
	return []models.SearchResult{
		{DocumentID: "c1", Category: "customer_data", Score: 0.9},
		{DocumentID: "m1", Category: "manufacturing_data", Score: 0.8},
		{DocumentID: "h1", Category: "hr_data", Score: 0.7},
	}
}

func run(t *testing.T, v Vault, req models.SearchRequest) (models.SearchResponse, *fakeEmbedder) {
	t.Helper()
	emb := &fakeEmbedder{}
	cfg := config.Defaults()
	o := New(cfg, emb, &fakeSearcher{results: docs()}, fakeReranker{}, v)
	resp, err := o.Search(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	return resp, emb
}

func ids(r models.SearchResponse) []string {
	var out []string
	for _, x := range r.Results {
		out = append(out, x.DocumentID)
	}
	return out
}

func TestSearchOnlyReturnsCategoriesVaultAllows(t *testing.T) {
	v := &fakeVault{cats: []string{"customer_data", "hr_data"}}
	resp, _ := run(t, v, models.SearchRequest{UserID: "u", TenantID: "t", Department: "sales", Roles: []string{"analyst"}, Query: "q"})
	if got := ids(resp); len(got) != 2 || got[0] != "c1" || got[1] != "h1" {
		t.Errorf("results = %v, want only customer_data and hr_data documents", got)
	}
	if resp.Metadata.FilteredOut != 1 {
		t.Errorf("FilteredOut = %d, want 1", resp.Metadata.FilteredOut)
	}
}

func TestVaultIsAskedAboutTheRequestsUser(t *testing.T) {
	v := &fakeVault{cats: []string{"customer_data"}}
	run(t, v, models.SearchRequest{UserID: "u1", TenantID: "t1", Department: "sales", Roles: []string{"analyst", "lead"}, Query: "q"})
	if v.got.UserID != "u1" || v.got.TenantID != "t1" || v.got.Department != "sales" || len(v.got.Roles) != 2 {
		t.Errorf("Vault was asked about %+v, want the request's user, tenant, department and roles", v.got)
	}
}

func TestNoAccessSearchesNothing(t *testing.T) {
	v := &fakeVault{cats: []string{}}
	resp, emb := run(t, v, models.SearchRequest{UserID: "u", Query: "q"})
	if len(resp.Results) != 0 || resp.Metadata.Strategy != "denied" {
		t.Errorf("got %v (%s), want no results and strategy denied", ids(resp), resp.Metadata.Strategy)
	}
	if emb.calls != 0 {
		t.Error("a denied request must not reach the embedder or the vector store")
	}
}

func TestVaultFailureDeniesInsteadOfAllowingEverything(t *testing.T) {
	v := &fakeVault{err: errors.New("vault down")}
	resp, _ := run(t, v, models.SearchRequest{UserID: "u", Department: "d", Roles: []string{"r"}, Query: "q"})
	if len(resp.Results) != 0 {
		t.Errorf("results = %v, want none when Vault cannot be consulted", ids(resp))
	}
}

func TestCallerCategoriesCanOnlyNarrowNeverGrant(t *testing.T) {
	v := &fakeVault{cats: []string{"customer_data"}}
	resp, _ := run(t, v, models.SearchRequest{
		UserID: "u", Query: "q",
		AllowedCategories: []string{"customer_data", "hr_data"}, // hr_data is not Vault-allowed
	})
	if got := ids(resp); len(got) != 1 || got[0] != "c1" {
		t.Errorf("results = %v, want only c1: a caller must not widen Vault's answer", got)
	}
	resp, _ = run(t, v, models.SearchRequest{UserID: "u", Query: "q", AllowedCategories: []string{"hr_data"}})
	if len(resp.Results) != 0 {
		t.Errorf("results = %v, want none when the caller asks only for a category Vault denies", ids(resp))
	}
}

func TestVaultDisabledIsUnrestrictedButCallerMayStillNarrow(t *testing.T) {
	resp, _ := run(t, vault.NewMock(), models.SearchRequest{Query: "q"})
	if len(resp.Results) != 3 {
		t.Errorf("results = %v, want everything when Vault is disabled", ids(resp))
	}
	resp, _ = run(t, vault.NewMock(), models.SearchRequest{Query: "q", AllowedCategories: []string{"hr_data"}})
	if got := ids(resp); len(got) != 1 || got[0] != "h1" {
		t.Errorf("results = %v, want only the category the caller restricted itself to", got)
	}
}
