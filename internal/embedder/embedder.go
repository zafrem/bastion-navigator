// Package embedder provides text-to-vector embedding implementations.
package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/bastion/navigator/internal/cache"
	"github.com/bastion/navigator/internal/config"
)

// Embedder converts text to dense float32 vectors.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// --- BGE (HTTP endpoint) ---

type bgeEmbedder struct {
	cfg    config.EmbedderConfig
	cache  cache.Cache
	client *http.Client
}

// NewBGE returns an Embedder that calls an HTTP embedding service
// (BGE-M3 or any OpenAI-compatible /embed endpoint).
func NewBGE(cfg config.EmbedderConfig, c cache.Cache) Embedder {
	return &bgeEmbedder{
		cfg:    cfg,
		cache:  c,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

func (e *bgeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if e.cache != nil {
		if v, ok := e.cache.Get(text); ok {
			var vec []float32
			if err := json.Unmarshal(v, &vec); err == nil {
				return vec, nil
			}
		}
	}
	vec, err := e.callEndpoint(ctx, []string{text})
	if err != nil || len(vec) == 0 {
		return nil, fmt.Errorf("embed: %w", err)
	}
	if e.cache != nil {
		if b, err := json.Marshal(vec[0]); err == nil {
			e.cache.Set(text, b)
		}
	}
	return vec[0], nil
}

func (e *bgeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return e.callEndpoint(ctx, texts)
}

type embedRequest struct {
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Embeddings [][]float32 `json:"embeddings"` // alternative schema
}

func (e *bgeEmbedder) callEndpoint(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Input: texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.cfg.Endpoint+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed endpoint %s: HTTP %d", e.cfg.Endpoint, resp.StatusCode)
	}
	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Embeddings) > 0 {
		return result.Embeddings, nil
	}
	out := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// --- Mock ---

type mockEmbedder struct {
	dim int
	rng *rand.Rand
}

// NewMock returns an Embedder that produces random unit vectors of the given dimensionality.
func NewMock(dim int) Embedder {
	return &mockEmbedder{dim: dim, rng: rand.New(rand.NewSource(42))}
}

func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return m.randomVec(), nil
}

func (m *mockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = m.randomVec()
	}
	return out, nil
}

func (m *mockEmbedder) randomVec() []float32 {
	vec := make([]float32, m.dim)
	var norm float32
	for i := range vec {
		v := float32(m.rng.NormFloat64())
		vec[i] = v
		norm += v * v
	}
	if norm > 0 {
		norm = float32(1.0 / float64(norm))
		for i := range vec {
			vec[i] *= norm
		}
	}
	return vec
}
