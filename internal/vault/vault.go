// Package vault provides a client for the Bastion Vault permission service.
package vault

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/zafrem/bastion-navigator/internal/cache"
	"github.com/zafrem/bastion-navigator/internal/models"
)

// Client queries Vault for user permissions and filters search results.
type Client interface {
	AllowedCategories(ctx context.Context, userID string) ([]string, error)
	FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int)
}

// --- HTTP client ---

type httpClient struct {
	endpoint  string
	cache     cache.Cache
	ttl       time.Duration
	http      *http.Client
	token     string
	tokenFile string
}

// Option configures the HTTP client.
type Option func(*httpClient)

// WithToken sets the service JWT sent as "Authorization: Bearer <token>".
func WithToken(token string) Option { return func(c *httpClient) { c.token = token } }

// WithTokenFile reads the service JWT from a file on every call, so rotation needs no restart.
// It takes precedence over WithToken; an unreadable or empty file is an error (Vault is not called).
func WithTokenFile(path string) Option { return func(c *httpClient) { c.tokenFile = path } }

// NewHTTP returns a Vault Client that calls the Bastion Vault REST API.
func NewHTTP(endpoint string, c cache.Cache, permTTL time.Duration, opts ...Option) Client {
	h := &httpClient{
		endpoint: strings.TrimRight(endpoint, "/"),
		cache:    c,
		ttl:      permTTL,
		http:     &http.Client{Timeout: 5 * time.Second},
	}
	for _, o := range opts {
		o(h)
	}
	return h
}

// authToken returns the current service token ("" when none is configured).
func (v *httpClient) authToken() (string, error) {
	if v.tokenFile != "" {
		b, err := os.ReadFile(v.tokenFile)
		if err != nil {
			return "", fmt.Errorf("vault: cannot read token file: %w", err)
		}
		tok := strings.TrimSpace(string(b))
		if tok == "" {
			return "", fmt.Errorf("vault: token file is empty")
		}
		return tok, nil
	}
	return v.token, nil
}

// Vault reports data categories as IDs; Navigator names them by content.
var vaultCategoryNames = map[string]string{
	"DC-01": "customer_data",
	"DC-02": "manufacturing_data",
	"DC-03": "hr_data",
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
	token, err := v.authToken()
	if err != nil {
		return nil, err
	}
	// Note: with only a user ID, Vault answers using the *token's* own department and role.
	endpoint := fmt.Sprintf("%s/v1/vault/permissions/%s", v.endpoint, url.PathEscape(userID))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
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
		Allowed *[]string `json:"allowed_categories"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if result.Allowed == nil {
		return nil, fmt.Errorf("vault AllowedCategories: response has no allowed_categories")
	}
	cats := make([]string, 0, len(*result.Allowed))
	for _, id := range *result.Allowed {
		if name, ok := vaultCategoryNames[id]; ok {
			id = name
		}
		cats = append(cats, id)
	}
	if v.cache != nil {
		if b, err := json.Marshal(cats); err == nil {
			v.cache.Set(cacheKey, b)
		}
	}
	return cats, nil
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
