package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/bastion/navigator/internal/cache"
	"github.com/bastion/navigator/internal/config"
	"github.com/bastion/navigator/internal/metrics"
)

// Embedder converts text into dense vectors.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// BGEClient calls the self-hosted BGE-M3 FastAPI service.
type BGEClient struct {
	endpoint  string
	httpClient *http.Client
	cache     cache.Cache
	cacheTTL  time.Duration
}

func NewBGE(cfg config.EmbedderConfig, c cache.Cache) *BGEClient {
	return &BGEClient{
		endpoint:  cfg.Endpoint,
		cache:     c,
		cacheTTL:  cfg.Cache.TTL,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

type embedRequest struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Embedding []float32 `json:"embedding"`
}

type batchEmbedRequest struct {
	Texts []string `json:"texts"`
}

type batchEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (b *BGEClient) Embed(ctx context.Context, text string) ([]float32, error) {
	if b.cache != nil {
		if vec, ok := b.cache.GetEmbedding(ctx, text); ok {
			metrics.CacheHits.Add(1)
			return vec, nil
		}
		metrics.CacheMisses.Add(1)
	}

	start := time.Now()
	vec, err := b.callEmbed(ctx, text)
	metrics.EmbeddingDuration.Observe(time.Since(start).Seconds())
	if err != nil {
		return nil, err
	}
	metrics.EmbeddingsGenerated.Add(1)

	if b.cache != nil {
		_ = b.cache.SetEmbedding(ctx, text, vec, b.cacheTTL)
	}
	return vec, nil
}

func (b *BGEClient) callEmbed(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(embedRequest{Text: text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.endpoint+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedder call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedder status %d", resp.StatusCode)
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}
	return result.Embedding, nil
}

func (b *BGEClient) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	start := time.Now()
	defer func() { metrics.EmbeddingDuration.Observe(time.Since(start).Seconds()) }()

	body, _ := json.Marshal(batchEmbedRequest{Texts: texts})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.endpoint+"/embed/batch", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedder batch call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedder returned status %d", resp.StatusCode)
	}

	var result batchEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode batch embed response: %w", err)
	}

	metrics.EmbeddingsGenerated.Add(float64(len(texts)))
	return result.Embeddings, nil
}

// MockEmbedder returns random L2-normalised vectors for standalone/testing mode.
type MockEmbedder struct {
	dims int
}

func NewMock(dims int) *MockEmbedder {
	if dims == 0 {
		dims = 1024
	}
	return &MockEmbedder{dims: dims}
}

func (m *MockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return randomNormalized(m.dims), nil
}

func (m *MockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = randomNormalized(m.dims)
	}
	return out, nil
}

func randomNormalized(dims int) []float32 {
	vec := make([]float32, dims)
	var sumSq float64
	for i := range vec {
		v := rand.Float32()*2 - 1
		vec[i] = v
		sumSq += float64(v) * float64(v)
	}
	if sumSq > 0 {
		norm := float32(1.0 / math.Sqrt(sumSq))
		for i := range vec {
			vec[i] *= norm
		}
	}
	return vec
}
