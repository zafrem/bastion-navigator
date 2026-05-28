// Package reranker provides cross-encoder re-ranking implementations.
package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/bastion/navigator/internal/models"
)

// Reranker re-scores candidate search results relative to a query.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error)
}

// --- BGE (HTTP endpoint) ---

type bgeReranker struct {
	endpoint string
	client   *http.Client
}

// NewBGE returns a Reranker that calls an HTTP cross-encoder endpoint.
func NewBGE(endpoint string) Reranker {
	return &bgeReranker{
		endpoint: endpoint,
		client:   &http.Client{Timeout: 30 * time.Second},
	}
}

type rerankRequest struct {
	Query      string   `json:"query"`
	Candidates []string `json:"candidates"`
}

type rerankResponse struct {
	Scores []float64 `json:"scores"`
}

func (r *bgeReranker) Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	texts := make([]string, len(candidates))
	for i, c := range candidates {
		texts[i] = c.Content
	}
	body, _ := json.Marshal(rerankRequest{Query: query, Candidates: texts})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank endpoint %s: HTTP %d", r.endpoint, resp.StatusCode)
	}
	var result rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	type scored struct {
		result models.SearchResult
		score  float64
	}
	items := make([]scored, len(candidates))
	for i, c := range candidates {
		score := c.Score
		if i < len(result.Scores) {
			score = result.Scores[i]
		}
		items[i] = scored{result: c, score: score}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].score > items[j].score })
	out := make([]models.SearchResult, len(items))
	for i, item := range items {
		out[i] = item.result
		out[i].Score = float64(item.score)
	}
	if topK > 0 && topK < len(out) {
		return out[:topK], nil
	}
	return out, nil
}

// --- Mock ---

type mockReranker struct{}

// NewMock returns a Reranker that returns candidates unchanged (sorted by original score).
func NewMock() Reranker {
	return &mockReranker{}
}

func (m *mockReranker) Rerank(_ context.Context, _ string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	sorted := make([]models.SearchResult, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
	if topK > 0 && topK < len(sorted) {
		return sorted[:topK], nil
	}
	return sorted, nil
}
