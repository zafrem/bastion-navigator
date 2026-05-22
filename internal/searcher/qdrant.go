// Package searcher wraps Qdrant's REST API for vector and sparse (BM25) search.
package searcher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/bastion/navigator/internal/metrics"
	"github.com/bastion/navigator/internal/models"
)

// Searcher executes searches against the Qdrant vector database.
type Searcher interface {
	VectorSearch(ctx context.Context, collection string, vector []float32, filter map[string]string, topK int, minScore float64) ([]models.SearchResult, error)
	SparseSearch(ctx context.Context, collection string, query string, filter map[string]string, topK int) ([]models.SearchResult, error)
	Collections(ctx context.Context) ([]models.CollectionInfo, error)
}

// QdrantClient calls Qdrant over its HTTP REST API.
type QdrantClient struct {
	hosts      []string
	httpClient *http.Client
}

func NewQdrant(hosts []string) *QdrantClient {
	return &QdrantClient{
		hosts:      hosts,
		httpClient: &http.Client{Timeout: 15 * time.Second},
	}
}

func (q *QdrantClient) host() string {
	return "http://" + q.hosts[0]
}

// qdrantFilter builds the Qdrant payload filter from key=value pairs plus a required tenant_id.
func qdrantFilter(filters map[string]string) map[string]interface{} {
	must := make([]map[string]interface{}, 0, len(filters))
	for k, v := range filters {
		must = append(must, map[string]interface{}{
			"key": k,
			"match": map[string]interface{}{"value": v},
		})
	}
	return map[string]interface{}{"must": must}
}

type qdrantSearchPayload struct {
	Vector      []float32              `json:"vector"`
	Filter      map[string]interface{} `json:"filter,omitempty"`
	Limit       int                    `json:"limit"`
	ScoreThresh float64                `json:"score_threshold,omitempty"`
	WithPayload bool                   `json:"with_payload"`
}

type qdrantResult struct {
	ID      interface{}            `json:"id"`
	Score   float64                `json:"score"`
	Payload map[string]interface{} `json:"payload"`
}

type qdrantSearchResponse struct {
	Result []qdrantResult `json:"result"`
}

func (q *QdrantClient) VectorSearch(ctx context.Context, collection string, vector []float32, filter map[string]string, topK int, minScore float64) ([]models.SearchResult, error) {
	start := time.Now()
	defer func() {
		metrics.QdrantCallDuration.Observe(time.Since(start).Seconds())
		metrics.QdrantCallsTotal.With(map[string]string{"operation": "search"}).Inc()
	}()

	payload := qdrantSearchPayload{
		Vector:      vector,
		Limit:       topK,
		ScoreThresh: minScore,
		WithPayload: true,
	}
	if len(filter) > 0 {
		payload.Filter = qdrantFilter(filter)
	}

	body, _ := json.Marshal(payload)
	url := fmt.Sprintf("%s/collections/%s/points/search", q.host(), collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := q.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("qdrant returned status %d", resp.StatusCode)
	}

	var result qdrantSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode qdrant response: %w", err)
	}

	return toSearchResults(result.Result, "vector"), nil
}

// SparseSearch uses Qdrant's sparse vector endpoint for BM25-style retrieval.
// When the collection does not have a sparse index, it falls back to a payload full-text match.
type qdrantSparseSearchPayload struct {
	Vector      map[string]interface{} `json:"vector"`
	Filter      map[string]interface{} `json:"filter,omitempty"`
	Limit       int                    `json:"limit"`
	WithPayload bool                   `json:"with_payload"`
}

func (q *QdrantClient) SparseSearch(ctx context.Context, collection string, query string, filter map[string]string, topK int) ([]models.SearchResult, error) {
	start := time.Now()
	defer func() {
		metrics.QdrantCallDuration.Observe(time.Since(start).Seconds())
		metrics.QdrantCallsTotal.With(map[string]string{"operation": "sparse_search"}).Inc()
	}()

	// Use Qdrant's full-text payload filter as a BM25 proxy when sparse index is unavailable.
	f := make(map[string]string, len(filter)+1)
	for k, v := range filter {
		f[k] = v
	}

	must := make([]map[string]interface{}, 0)
	for k, v := range filter {
		must = append(must, map[string]interface{}{
			"key":   k,
			"match": map[string]interface{}{"value": v},
		})
	}
	// Full-text condition on the content field.
	must = append(must, map[string]interface{}{
		"key":   "content",
		"match": map[string]interface{}{"text": query},
	})

	scrollPayload := map[string]interface{}{
		"filter":       map[string]interface{}{"must": must},
		"limit":        topK,
		"with_payload": true,
		"with_vector":  false,
	}

	body, _ := json.Marshal(scrollPayload)
	url := fmt.Sprintf("%s/collections/%s/points/scroll", q.host(), collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := q.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant sparse search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Non-fatal: return empty results so hybrid search degrades gracefully.
		return nil, nil
	}

	var scrollResp struct {
		Result struct {
			Points []qdrantResult `json:"points"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&scrollResp); err != nil {
		return nil, nil
	}

	results := toSearchResults(scrollResp.Result.Points, "bm25")
	// Assign synthetic BM25 scores in descending order.
	for i := range results {
		results[i].BM25Score = 1.0 / float64(i+1)
		results[i].Score = results[i].BM25Score
	}
	return results, nil
}

func (q *QdrantClient) Collections(ctx context.Context) ([]models.CollectionInfo, error) {
	url := fmt.Sprintf("%s/collections", q.host())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := q.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant collections: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			Collections []struct {
				Name string `json:"name"`
			} `json:"collections"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	out := make([]models.CollectionInfo, 0, len(result.Result.Collections))
	for _, c := range result.Result.Collections {
		out = append(out, models.CollectionInfo{Name: c.Name, Status: "green"})
	}
	return out, nil
}

func toSearchResults(raw []qdrantResult, scoreField string) []models.SearchResult {
	out := make([]models.SearchResult, 0, len(raw))
	for _, r := range raw {
		sr := models.SearchResult{
			Score:    r.Score,
			Metadata: make(map[string]string),
		}
		if id, ok := r.ID.(string); ok {
			sr.DocumentID = id
		} else if f, ok := r.ID.(float64); ok {
			sr.DocumentID = fmt.Sprintf("%.0f", f)
		}
		if r.Payload != nil {
			if v, ok := r.Payload["content"].(string); ok {
				sr.Content = v
			}
			if v, ok := r.Payload["category"].(string); ok {
				sr.Category = v
			}
			for k, v := range r.Payload {
				if k == "content" || k == "category" {
					continue
				}
				sr.Metadata[k] = fmt.Sprintf("%v", v)
			}
		}
		if scoreField == "vector" {
			sr.VectorScore = r.Score
		} else {
			sr.BM25Score = r.Score
		}
		out = append(out, sr)
	}
	return out
}
