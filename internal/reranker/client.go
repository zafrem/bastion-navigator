package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/bastion/navigator/internal/metrics"
	"github.com/bastion/navigator/internal/models"
)

// Reranker re-scores a candidate list using a cross-encoder model.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error)
}

// BGERerankerClient calls the self-hosted BGE-reranker-v2-m3 FastAPI service.
type BGERerankerClient struct {
	endpoint   string
	httpClient *http.Client
}

func NewBGE(endpoint string) *BGERerankerClient {
	return &BGERerankerClient{
		endpoint:   endpoint,
		httpClient: &http.Client{Timeout: 15 * time.Second},
	}
}

type rerankRequest struct {
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
}

type rerankResponse struct {
	Scores []float64 `json:"scores"`
}

func (b *BGERerankerClient) Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	if len(candidates) < 5 {
		return candidates, nil
	}

	start := time.Now()
	defer func() {
		metrics.RerankDuration.Observe(time.Since(start).Seconds())
		metrics.RerankingsTotal.Add(1)
	}()

	passages := make([]string, len(candidates))
	for i, c := range candidates {
		passages[i] = c.Content
	}

	body, _ := json.Marshal(rerankRequest{Query: query, Passages: passages})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.endpoint+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("reranker call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("reranker status %d", resp.StatusCode)
	}

	var result rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}
	if len(result.Scores) != len(candidates) {
		return nil, fmt.Errorf("reranker returned %d scores for %d candidates", len(result.Scores), len(candidates))
	}

	out := make([]models.SearchResult, len(candidates))
	copy(out, candidates)
	for i := range out {
		out[i].RerankScore = result.Scores[i]
		out[i].Score = result.Scores[i]
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].RerankScore > out[j].RerankScore
	})

	if topK > 0 && topK < len(out) {
		out = out[:topK]
	}
	return out, nil
}

// MockReranker reverses the candidate list to simulate reranking without a model.
type MockReranker struct{}

func NewMock() *MockReranker { return &MockReranker{} }

func (m *MockReranker) Rerank(_ context.Context, _ string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	out := make([]models.SearchResult, len(candidates))
	copy(out, candidates)
	for i := range out {
		out[i].RerankScore = out[i].Score * 0.95
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].RerankScore > out[j].RerankScore
	})
	if topK > 0 && topK < len(out) {
		out = out[:topK]
	}
	return out, nil
}
