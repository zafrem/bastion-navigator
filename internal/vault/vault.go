// Package vault provides a client for the Bastion Vault permission service.
package vault

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/bastion/navigator/internal/cache"
	"github.com/bastion/navigator/internal/models"
)

// Client queries Vault for user permissions and filters search results.
type Client interface {
	AllowedCategories(ctx context.Context, userID string) ([]string, error)
	FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int)
}

// --- HTTP client ---

type httpClient struct {
	endpoint string
	cache    cache.Cache
	ttl      time.Duration
	http     *http.Client
}

// NewHTTP returns a Vault Client that calls the Bastion Vault REST API.
func NewHTTP(endpoint string, c cache.Cache, permTTL time.Duration) Client {
	return &httpClient{
		endpoint: endpoint,
		cache:    c,
		ttl:      permTTL,
		http:     &http.Client{Timeout: 5 * time.Second},
	}
}

func (v *httpClient) AllowedCategories(ctx context.Context, userID string) ([]string, error) {
	cacheKey := "vault:cats:" + userID
	if v.cache != nil {
		if b, ok := v.cache.Get(cacheKey); ok {
			var cats []string
			if err := json.Unmarshal(b, &cats); err == nil {
				return cats, nil
			}
		}
	}
	url := fmt.Sprintf("%s/v1/vault/access/categories?user_id=%s", v.endpoint, userID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := v.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vault AllowedCategories: HTTP %d", resp.StatusCode)
	}
	var result struct {
		Categories []string `json:"categories"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if v.cache != nil {
		if b, err := json.Marshal(result.Categories); err == nil {
			v.cache.Set(cacheKey, b)
		}
	}
	return result.Categories, nil
}

func (v *httpClient) FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	return filterByCategorySet(results, categories)
}

// --- Mock ---

type mockClient struct{}

// NewMock returns a Vault Client that allows all results through (no permission filtering).
func NewMock() Client {
	return &mockClient{}
}

func (m *mockClient) AllowedCategories(_ context.Context, _ string) ([]string, error) {
	return nil, nil
}

func (m *mockClient) FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	return filterByCategorySet(results, categories)
}

func filterByCategorySet(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	if len(categories) == 0 {
		return results, 0
	}
	allowed := make(map[string]struct{}, len(categories))
	for _, c := range categories {
		allowed[c] = struct{}{}
	}
	out := results[:0]
	filtered := 0
	for _, r := range results {
		if _, ok := allowed[r.Category]; ok {
			out = append(out, r)
		} else {
			filtered++
		}
	}
	return out, filtered
}
