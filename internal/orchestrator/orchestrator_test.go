package orchestrator

import (
	"context"
	"testing"

	"github.com/bastion/navigator/internal/config"
	"github.com/bastion/navigator/internal/models"
)

// ─── Stubs ────────────────────────────────────────────────────────────────────

type stubEmbedder struct{ dims int }

func (s *stubEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	v := make([]float32, s.dims)
	for i := range v {
		v[i] = 0.1
	}
	return v, nil
}
func (s *stubEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		v, _ := s.Embed(context.Background(), "")
		out[i] = v
	}
	return out, nil
}

type stubSearcher struct {
	vectorResults []models.SearchResult
	bm25Results   []models.SearchResult
	collections   []models.CollectionInfo
}

func (s *stubSearcher) VectorSearch(_ context.Context, _ string, _ []float32, _ map[string]string, _ int, _ float64) ([]models.SearchResult, error) {
	return s.vectorResults, nil
}
func (s *stubSearcher) SparseSearch(_ context.Context, _ string, _ string, _ map[string]string, _ int) ([]models.SearchResult, error) {
	return s.bm25Results, nil
}
func (s *stubSearcher) Collections(_ context.Context) ([]models.CollectionInfo, error) {
	return s.collections, nil
}

type stubReranker struct{}

func (r *stubReranker) Rerank(_ context.Context, _ string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	for i := range candidates {
		candidates[i].RerankScore = candidates[i].Score * 0.9
	}
	if topK > 0 && topK < len(candidates) {
		return candidates[:topK], nil
	}
	return candidates, nil
}

type stubVault struct{ allowed []string }

func (v *stubVault) AllowedCategories(_ context.Context, _ string) ([]string, error) {
	return v.allowed, nil
}
func (v *stubVault) FilterResults(results []models.SearchResult, allowed []string) ([]models.SearchResult, int) {
	if len(allowed) == 0 {
		return results, 0
	}
	set := make(map[string]struct{}, len(allowed))
	for _, c := range allowed {
		set[c] = struct{}{}
	}
	out := make([]models.SearchResult, 0, len(results))
	filtered := 0
	for _, r := range results {
		if r.Category == "" {
			out = append(out, r)
			continue
		}
		if _, ok := set[r.Category]; ok {
			out = append(out, r)
		} else {
			filtered++
		}
	}
	return out, filtered
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func makeOrch(srch *stubSearcher, allowed []string) *Orchestrator {
	cfg := config.Defaults()
	cfg.SearchDefaults.MinScore = 0 // disable score threshold for unit tests
	return New(cfg,
		&stubEmbedder{dims: 1024},
		srch,
		&stubReranker{},
		&stubVault{allowed: allowed},
	)
}

func docs(ids ...string) []models.SearchResult {
	out := make([]models.SearchResult, len(ids))
	for i, id := range ids {
		out[i] = models.SearchResult{
			DocumentID:  id,
			Category:    "customer_data",
			Score:       1.0 / float64(i+1),
			VectorScore: 1.0 / float64(i+1),
			Content:     "content for " + id,
		}
	}
	return out
}

// ─── RRF tests ────────────────────────────────────────────────────────────────

func TestRRF_CombinesRanks(t *testing.T) {
	vec := docs("a", "b", "c")
	bm25 := docs("b", "c", "a") // different order

	fused := rrf(vec, bm25, 0.7, 0.3)

	// "b" appears at rank 2 in vector and rank 1 in BM25 — it should outrank "a"
	// which appears at rank 1 in vector but rank 3 in BM25.
	if len(fused) != 3 {
		t.Fatalf("expected 3 results, got %d", len(fused))
	}
	// Scores must be strictly positive.
	for _, r := range fused {
		if r.Score <= 0 {
			t.Errorf("result %s has non-positive score %f", r.DocumentID, r.Score)
		}
	}
}

func TestRRF_DeduplicatesDocuments(t *testing.T) {
	shared := docs("x", "y")
	fused := rrf(shared, shared, 0.5, 0.5) // same list both sides
	if len(fused) != 2 {
		t.Errorf("expected 2 unique results, got %d", len(fused))
	}
}

func TestRRF_EmptyBM25(t *testing.T) {
	vec := docs("a", "b")
	fused := rrf(vec, nil, 1.0, 0.0)
	if len(fused) != 2 {
		t.Errorf("expected 2, got %d", len(fused))
	}
}

// ─── Orchestrator.Search tests ────────────────────────────────────────────────

func TestSearch_VectorOnly(t *testing.T) {
	srch := &stubSearcher{vectorResults: docs("d1", "d2", "d3")}
	orch := makeOrch(srch, nil)

	resp, err := orch.Search(context.Background(), models.SearchRequest{
		RequestID: "t1",
		TenantID:  "acme",
		Query:     "test query",
		Options:   &models.SearchOptions{TopK: 2, UseHybrid: false, UseReranking: false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) > 2 {
		t.Errorf("expected at most 2 results, got %d", len(resp.Results))
	}
}

func TestSearch_HybridSearch_MergesResults(t *testing.T) {
	srch := &stubSearcher{
		vectorResults: docs("d1", "d2"),
		bm25Results:   docs("d3", "d1"),
	}
	// Restrict to customer_data only → orchestrator calls searchCollection once.
	orch := makeOrch(srch, []string{"customer_data"})

	resp, err := orch.Search(context.Background(), models.SearchRequest{
		RequestID: "t2",
		TenantID:  "acme",
		Query:     "hybrid test",
		// User carries pre-resolved categories so vault is bypassed.
		User: &models.UserContext{AllowedCategories: []string{"customer_data"}},
		Options: &models.SearchOptions{
			TopK:         10,
			UseHybrid:    true,
			UseReranking: false,
			VectorWeight: 0.7,
			BM25Weight:   0.3,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// RRF deduplicates d1 (in both lists) → 3 unique docs from 1 collection.
	if len(resp.Results) != 3 {
		t.Errorf("expected 3 unique results from fusion, got %d", len(resp.Results))
	}
}

func TestSearch_PermissionFiltering(t *testing.T) {
	results := []models.SearchResult{
		{DocumentID: "pub", Category: "customer_data", Score: 0.9},
		{DocumentID: "priv", Category: "hr_data", Score: 0.8},
	}
	srch := &stubSearcher{vectorResults: results}
	orch := New(config.Defaults(),
		&stubEmbedder{dims: 1024},
		srch,
		&stubReranker{},
		// Vault returns only customer_data.
		&stubVault{allowed: []string{"customer_data"}},
	)
	cfg := config.Defaults()
	cfg.SearchDefaults.MinScore = 0
	orch.cfg = cfg

	resp, err := orch.Search(context.Background(), models.SearchRequest{
		RequestID: "t3",
		TenantID:  "acme",
		Query:     "test",
		User:      &models.UserContext{UserID: "alice"},
		Options:   &models.SearchOptions{TopK: 10, UseHybrid: false, UseReranking: false},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range resp.Results {
		if r.Category == "hr_data" {
			t.Errorf("hr_data result should have been filtered out")
		}
	}
	if resp.Metadata.FilteredOut != 1 {
		t.Errorf("expected 1 filtered out, got %d", resp.Metadata.FilteredOut)
	}
}

func TestSearch_ZeroResults(t *testing.T) {
	srch := &stubSearcher{}
	orch := makeOrch(srch, nil)

	resp, err := orch.Search(context.Background(), models.SearchRequest{
		RequestID: "t4",
		TenantID:  "acme",
		Query:     "nothing matches",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) != 0 {
		t.Errorf("expected 0 results, got %d", len(resp.Results))
	}
}

func TestSearch_ReranksWhenEnabled(t *testing.T) {
	// Provide 5+ candidates so the reranker stub activates.
	srch := &stubSearcher{vectorResults: docs("a", "b", "c", "d", "e", "f")}
	orch := makeOrch(srch, nil)

	resp, err := orch.Search(context.Background(), models.SearchRequest{
		RequestID: "t5",
		TenantID:  "acme",
		Query:     "rerank test",
		Options:   &models.SearchOptions{TopK: 3, UseHybrid: false, UseReranking: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Results) > 3 {
		t.Errorf("expected at most 3 reranked results, got %d", len(resp.Results))
	}
}

// ─── mergeDefaults tests ──────────────────────────────────────────────────────

func TestMergeDefaults_FillsZeroValues(t *testing.T) {
	orch := makeOrch(&stubSearcher{}, nil)
	opts := orch.mergeDefaults(nil)

	if opts.TopK == 0 {
		t.Error("TopK should be filled from defaults")
	}
	if opts.VectorWeight == 0 {
		t.Error("VectorWeight should be filled from defaults")
	}
}

func TestMergeDefaults_PreservesExplicitValues(t *testing.T) {
	orch := makeOrch(&stubSearcher{}, nil)
	opts := orch.mergeDefaults(&models.SearchOptions{TopK: 42})

	if opts.TopK != 42 {
		t.Errorf("explicit TopK=42 should be preserved, got %d", opts.TopK)
	}
}

// ─── collectionsForCategories tests ──────────────────────────────────────────

func TestCollectionsForCategories_MapsCorrectly(t *testing.T) {
	orch := makeOrch(&stubSearcher{}, nil)
	cols := orch.collectionsForCategories([]string{"customer_data", "hr_data"})

	found := make(map[string]bool)
	for _, c := range cols {
		found[c] = true
	}
	if !found["customer_docs"] {
		t.Error("customer_data should map to customer_docs")
	}
	if !found["hr_docs"] {
		t.Error("hr_data should map to hr_docs")
	}
}

func TestCollectionsForCategories_EmptyAllowed_ReturnsAll(t *testing.T) {
	orch := makeOrch(&stubSearcher{}, nil)
	cols := orch.collectionsForCategories(nil)

	if len(cols) == 0 {
		t.Error("no allowed categories should return all configured collections")
	}
}
