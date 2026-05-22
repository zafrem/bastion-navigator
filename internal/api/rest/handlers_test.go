package rest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/bastion/navigator/internal/config"
	"github.com/bastion/navigator/internal/models"
	"github.com/bastion/navigator/internal/orchestrator"
)

// ─── Stubs reused from orchestrator tests ─────────────────────────────────────

type noopEmbedder struct{}

func (noopEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}
func (noopEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{0.1, 0.2, 0.3}
	}
	return out, nil
}

type noopSearcher struct{}

func (noopSearcher) VectorSearch(_ context.Context, _ string, _ []float32, _ map[string]string, _ int, _ float64) ([]models.SearchResult, error) {
	return []models.SearchResult{
		{DocumentID: "doc-1", Content: "hello world", Score: 0.9, Category: "customer_data"},
	}, nil
}
func (noopSearcher) SparseSearch(_ context.Context, _ string, _ string, _ map[string]string, _ int) ([]models.SearchResult, error) {
	return nil, nil
}
func (noopSearcher) Collections(_ context.Context) ([]models.CollectionInfo, error) {
	return []models.CollectionInfo{
		{Name: "customer_docs", VectorCount: 1000, Status: "green"},
	}, nil
}

type noopReranker struct{}

func (noopReranker) Rerank(_ context.Context, _ string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	if topK > 0 && topK < len(candidates) {
		return candidates[:topK], nil
	}
	return candidates, nil
}

type openVault struct{}

func (openVault) AllowedCategories(_ context.Context, _ string) ([]string, error) { return nil, nil }
func (openVault) FilterResults(r []models.SearchResult, _ []string) ([]models.SearchResult, int) {
	return r, 0
}

func testServer() *Server {
	cfg := config.Defaults()
	cfg.SearchDefaults.MinScore = 0
	orch := orchestrator.New(cfg, noopEmbedder{}, noopSearcher{}, noopReranker{}, openVault{})
	return New(orch, 0)
}

func post(t *testing.T, srv *Server, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.routes().ServeHTTP(w, req)
	return w
}

func get(t *testing.T, srv *Server, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	srv.routes().ServeHTTP(w, req)
	return w
}

// ─── Health tests ─────────────────────────────────────────────────────────────

func TestHealth_Returns200(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/health")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHealthLive_Returns200(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/health/live")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHealthReady_Returns200(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/health/ready")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

// ─── Search tests ─────────────────────────────────────────────────────────────

func TestSearch_ValidRequest_Returns200WithResults(t *testing.T) {
	srv := testServer()
	req := models.SearchRequest{
		RequestID: "test-1",
		TenantID:  "acme",
		Query:     "hello",
	}
	w := post(t, srv, "/v1/navigator/search", req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp models.SearchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.RequestID != "test-1" {
		t.Errorf("request_id not echoed: %s", resp.RequestID)
	}
	if len(resp.Results) == 0 {
		t.Error("expected at least one result")
	}
}

func TestSearch_InvalidJSON_Returns400(t *testing.T) {
	srv := testServer()
	req := httptest.NewRequest(http.MethodPost, "/v1/navigator/search", bytes.NewBufferString("{bad json}"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.routes().ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestHybridSearch_ForcesHybridFlag(t *testing.T) {
	srv := testServer()
	req := models.SearchRequest{
		RequestID: "test-hybrid",
		TenantID:  "acme",
		Query:     "hybrid test",
		Options:   &models.SearchOptions{UseHybrid: false},
	}
	w := post(t, srv, "/v1/navigator/search/hybrid", req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestBatchSearch_ReturnsMultipleResponses(t *testing.T) {
	srv := testServer()
	batch := models.BatchSearchRequest{
		RequestID: "batch-1",
		Queries: []models.SearchRequest{
			{RequestID: "q1", TenantID: "acme", Query: "first"},
			{RequestID: "q2", TenantID: "acme", Query: "second"},
		},
	}
	w := post(t, srv, "/v1/navigator/search/batch", batch)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp models.BatchSearchResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Errorf("expected 2 batch results, got %d", len(resp.Results))
	}
}

// ─── Embed tests ──────────────────────────────────────────────────────────────

func TestEmbed_Returns200WithVector(t *testing.T) {
	srv := testServer()
	w := post(t, srv, "/v1/navigator/embed", models.EmbedRequest{
		RequestID: "e1",
		Text:      "hello world",
	})
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp models.EmbedResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Embedding) == 0 {
		t.Error("expected non-empty embedding")
	}
	if resp.DimCount != len(resp.Embedding) {
		t.Errorf("dim_count mismatch: %d vs len %d", resp.DimCount, len(resp.Embedding))
	}
}

func TestBatchEmbed_ReturnsCorrectCount(t *testing.T) {
	srv := testServer()
	w := post(t, srv, "/v1/navigator/embed/batch", models.BatchEmbedRequest{
		Texts: []string{"first", "second", "third"},
	})
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var resp models.BatchEmbedResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Embeddings) != 3 {
		t.Errorf("expected 3 embeddings, got %d", len(resp.Embeddings))
	}
}

// ─── Rerank tests ─────────────────────────────────────────────────────────────

func TestRerank_Returns200(t *testing.T) {
	srv := testServer()
	w := post(t, srv, "/v1/navigator/rerank", models.RerankRequest{
		Query: "test",
		Candidates: []models.SearchResult{
			{DocumentID: "a", Content: "apple", Score: 0.9},
			{DocumentID: "b", Content: "banana", Score: 0.8},
		},
		TopK: 1,
	})
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

// ─── Collections tests ────────────────────────────────────────────────────────

func TestCollections_Returns200WithList(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/navigator/collections")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var resp models.CollectionsResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Collections) == 0 {
		t.Error("expected at least one collection")
	}
}

func TestCollectionInfo_KnownCollection_Returns200(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/navigator/collections/customer_docs")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 for known collection, got %d", w.Code)
	}
}

func TestCollectionInfo_UnknownCollection_Returns404(t *testing.T) {
	srv := testServer()
	w := get(t, srv, "/v1/navigator/collections/nonexistent")
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}
