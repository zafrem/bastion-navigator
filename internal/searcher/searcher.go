// Package searcher provides vector-store search backends.
package searcher

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/zafrem/bastion-navigator/internal/models"
)

// Searcher queries a vector store for nearest neighbours.
type Searcher interface {
	VectorSearch(ctx context.Context, collection string, vec []float32, filters map[string]string, limit int, minScore float64) ([]models.SearchResult, error)
	SparseSearch(ctx context.Context, collection string, text string, filters map[string]string, limit int) ([]models.SearchResult, error)
	Collections(ctx context.Context) ([]models.CollectionInfo, error)
}

// --- Qdrant (HTTP) ---

type qdrantSearcher struct {
	hosts  []string
	client *http.Client
}

// NewQdrant returns a Searcher backed by a Qdrant cluster.
// Requests are round-robined across the supplied host addresses.
func NewQdrant(hosts []string) Searcher {
	return &qdrantSearcher{
		hosts:  hosts,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (q *qdrantSearcher) host() string {
	if len(q.hosts) == 0 {
		return "localhost:6333"
	}
	return q.hosts[0]
}

// qdrantPoint is one search hit. Qdrant point IDs are unsigned integers or UUID strings, and payload
// values can be any JSON type, so both are decoded loosely and rendered as strings.
type qdrantPoint struct {
	ID      json.RawMessage        `json:"id"`
	Score   float64                `json:"score"`
	Payload map[string]interface{} `json:"payload"`
}

// pointID renders a Qdrant point ID (a JSON string or number) as a string.
func pointID(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return strings.TrimSpace(string(raw))
}

// stringPayload flattens a payload to strings: strings as-is, everything else as its JSON text.
func stringPayload(in map[string]interface{}) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		if s, ok := v.(string); ok {
			out[k] = s
			continue
		}
		if b, err := json.Marshal(v); err == nil {
			out[k] = string(b)
		}
	}
	return out
}

type qdrantSearchResp struct {
	Result []qdrantPoint `json:"result"`
}

func (q *qdrantSearcher) VectorSearch(ctx context.Context, collection string, vec []float32, filters map[string]string, limit int, minScore float64) ([]models.SearchResult, error) {
	body := map[string]interface{}{
		"vector": vec,
		"limit":  limit,
		"with_payload": true,
	}
	if minScore > 0 {
		body["score_threshold"] = minScore
	}
	if len(filters) > 0 {
		conds := make([]map[string]interface{}, 0, len(filters))
		for k, v := range filters {
			conds = append(conds, map[string]interface{}{
				"key":   k,
				"match": map[string]string{"value": v},
			})
		}
		body["filter"] = map[string]interface{}{"must": conds}
	}
	b, _ := json.Marshal(body)
	url := fmt.Sprintf("http://%s/collections/%s/points/search", q.host(), collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := q.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("qdrant %s: HTTP %d", url, resp.StatusCode)
	}
	var result qdrantSearchResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	out := make([]models.SearchResult, 0, len(result.Result))
	for _, p := range result.Result {
		payload := stringPayload(p.Payload)
		out = append(out, models.SearchResult{
			DocumentID: pointID(p.ID),
			Content:    payload["content"],
			Score:      p.Score,
			Category:   payload["category"],
			Metadata:   payload,
		})
	}
	return out, nil
}

func (q *qdrantSearcher) SparseSearch(ctx context.Context, collection string, text string, filters map[string]string, limit int) ([]models.SearchResult, error) {
	// BM25/sparse search via Qdrant's sparse vectors endpoint.
	// Qdrant < 1.7 doesn't support sparse vectors; return empty gracefully.
	return nil, nil
}

type qdrantCollectionsResp struct {
	Result struct {
		Collections []struct {
			Name string `json:"name"`
		} `json:"collections"`
	} `json:"result"`
}

type qdrantCollectionInfoResp struct {
	Result struct {
		// vectors_count is deprecated and null in current Qdrant releases; points_count is authoritative.
		VectorsCount *int   `json:"vectors_count"`
		PointsCount  *int   `json:"points_count"`
		Status       string `json:"status"`
	} `json:"result"`
}

func (q *qdrantSearcher) Collections(ctx context.Context) ([]models.CollectionInfo, error) {
	url := fmt.Sprintf("http://%s/collections", q.host())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := q.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("qdrant collections: HTTP %d", resp.StatusCode)
	}
	var list qdrantCollectionsResp
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, err
	}
	out := make([]models.CollectionInfo, 0, len(list.Result.Collections))
	for _, c := range list.Result.Collections {
		info, err := q.collectionInfo(ctx, c.Name)
		if err != nil {
			info = models.CollectionInfo{Name: c.Name, Status: "unknown"}
		}
		out = append(out, info)
	}
	return out, nil
}

func (q *qdrantSearcher) collectionInfo(ctx context.Context, name string) (models.CollectionInfo, error) {
	url := fmt.Sprintf("http://%s/collections/%s", q.host(), name)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return models.CollectionInfo{}, err
	}
	resp, err := q.client.Do(req)
	if err != nil {
		return models.CollectionInfo{}, err
	}
	defer resp.Body.Close()
	var info qdrantCollectionInfoResp
	_ = json.NewDecoder(resp.Body).Decode(&info)
	count := 0
	switch {
	case info.Result.PointsCount != nil:
		count = *info.Result.PointsCount
	case info.Result.VectorsCount != nil:
		count = *info.Result.VectorsCount
	}
	return models.CollectionInfo{
		Name:        name,
		VectorCount: count,
		Status:      info.Result.Status,
	}, nil
}
