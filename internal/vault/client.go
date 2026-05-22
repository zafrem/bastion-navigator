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

// Client resolves and caches per-user access permissions from Module B (Vault).
type Client interface {
	AllowedCategories(ctx context.Context, userID string) ([]string, error)
	FilterResults(results []models.SearchResult, allowed []string) ([]models.SearchResult, int)
}

// VaultHTTPClient contacts the live Vault service.
type VaultHTTPClient struct {
	endpoint   string
	httpClient *http.Client
	cache      cache.Cache
	cacheTTL   time.Duration
}

func NewHTTP(endpoint string, c cache.Cache, ttl time.Duration) *VaultHTTPClient {
	return &VaultHTTPClient{
		endpoint:   endpoint,
		httpClient: &http.Client{Timeout: 5 * time.Second},
		cache:      c,
		cacheTTL:   ttl,
	}
}

type permissionsResponse struct {
	UserID     string   `json:"user_id"`
	Categories []string `json:"allowed_categories"`
}

func (v *VaultHTTPClient) AllowedCategories(ctx context.Context, userID string) ([]string, error) {
	if v.cache != nil {
		if cats, ok := v.cache.GetPermissions(ctx, userID); ok {
			return cats, nil
		}
	}

	url := fmt.Sprintf("%s/v1/vault/permissions/%s", v.endpoint, userID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vault call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vault returned status %d", resp.StatusCode)
	}

	var pr permissionsResponse
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		return nil, err
	}

	if v.cache != nil {
		_ = v.cache.SetPermissions(ctx, userID, pr.Categories, v.cacheTTL)
	}
	return pr.Categories, nil
}

func (v *VaultHTTPClient) FilterResults(results []models.SearchResult, allowed []string) ([]models.SearchResult, int) {
	return filterByCategory(results, allowed)
}

// MockVaultClient grants access to all categories — used in standalone mode.
type MockVaultClient struct {
	AllCategories []string
}

func NewMock() *MockVaultClient {
	return &MockVaultClient{
		AllCategories: []string{"customer_data", "manufacturing_data", "hr_data"},
	}
}

func (m *MockVaultClient) AllowedCategories(_ context.Context, _ string) ([]string, error) {
	return m.AllCategories, nil
}

func (m *MockVaultClient) FilterResults(results []models.SearchResult, allowed []string) ([]models.SearchResult, int) {
	if len(allowed) == 0 {
		return results, 0
	}
	return filterByCategory(results, allowed)
}

func filterByCategory(results []models.SearchResult, allowed []string) ([]models.SearchResult, int) {
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
			// Uncategorised documents pass through.
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
