// Package vault provides a client for the Bastion Vault permission service.
package vault

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/zafrem/bastion-navigator/internal/cache"
	"github.com/zafrem/bastion-navigator/internal/models"
)

// Client resolves which data categories a user may search and filters results to them.
type Client interface {
	// AllowedCategories returns the categories the user may read. A nil slice means "unrestricted"
	// (Vault is disabled, so there is no permission model); a non-nil empty slice means no access.
	AllowedCategories(ctx context.Context, user models.UserContext) ([]string, error)
	// FilterResults keeps results whose category is in categories (nil keeps everything) and reports
	// how many were removed.
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
	now       func() time.Time
}

// Option configures the HTTP client.
type Option func(*httpClient)

// WithToken sets the service JWT sent as "Authorization: Bearer <token>".
func WithToken(token string) Option { return func(c *httpClient) { c.token = token } }

// WithTokenFile reads the service JWT from a file on every call, so rotation needs no restart.
// It takes precedence over WithToken; an unreadable or empty file is an error (Vault is not called).
func WithTokenFile(path string) Option { return func(c *httpClient) { c.tokenFile = path } }

// NewHTTP returns a Vault Client that calls the Bastion Vault REST API. Answers are cached for permTTL
// when c is non-nil and permTTL is positive; otherwise every call goes to Vault.
func NewHTTP(endpoint string, c cache.Cache, permTTL time.Duration, opts ...Option) Client {
	h := &httpClient{
		endpoint: strings.TrimRight(endpoint, "/"),
		cache:    c,
		ttl:      permTTL,
		http:     &http.Client{Timeout: 5 * time.Second},
		now:      time.Now,
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

type cachedCategories struct {
	Categories []string `json:"categories"`
	ExpiresAt  int64    `json:"expires_at"` // unix nanoseconds
}

func cacheKey(u models.UserContext, roles []string) string {
	sorted := append([]string(nil), roles...)
	sort.Strings(sorted)
	return "vault:cats:" + url.QueryEscape(u.TenantID) + "|" + url.QueryEscape(u.UserID) + "|" +
		url.QueryEscape(u.Department) + "|" + url.QueryEscape(strings.Join(sorted, ","))
}

func (v *httpClient) AllowedCategories(ctx context.Context, user models.UserContext) ([]string, error) {
	if user.UserID == "" {
		return []string{}, nil // no identity: nothing to look up, nothing to grant
	}
	roles := make([]string, 0, len(user.Roles))
	for _, r := range user.Roles {
		if r != "" {
			roles = append(roles, r)
		}
	}
	// Vault decides from department and role. Without them it would answer for the *service token's*
	// own department and role, which says nothing about this user, so deny without asking.
	if user.Department == "" || len(roles) == 0 {
		return []string{}, nil
	}

	key := cacheKey(user, roles)
	if v.cache != nil && v.ttl > 0 {
		if b, ok := v.cache.Get(key); ok {
			var hit cachedCategories
			if err := json.Unmarshal(b, &hit); err == nil && hit.Categories != nil && v.now().UnixNano() < hit.ExpiresAt {
				return hit.Categories, nil
			}
			v.cache.Delete(key)
		}
	}

	token, err := v.authToken()
	if err != nil {
		return nil, err
	}
	cats := []string{}
	seen := map[string]struct{}{}
	for _, role := range roles {
		got, err := v.fetch(ctx, user, role, token)
		if err != nil {
			return nil, err
		}
		for _, c := range got {
			if _, dup := seen[c]; !dup {
				seen[c] = struct{}{}
				cats = append(cats, c)
			}
		}
	}

	if v.cache != nil && v.ttl > 0 {
		if b, err := json.Marshal(cachedCategories{Categories: cats, ExpiresAt: v.now().Add(v.ttl).UnixNano()}); err == nil {
			v.cache.Set(key, b)
		}
	}
	return cats, nil
}

// fetch asks Vault which categories one department+role pair may read.
func (v *httpClient) fetch(ctx context.Context, user models.UserContext, role, token string) ([]string, error) {
	q := url.Values{}
	q.Set("department", user.Department)
	q.Set("role", role)
	if user.TenantID != "" {
		q.Set("tenant_id", user.TenantID)
	}
	endpoint := fmt.Sprintf("%s/v1/vault/permissions/%s?%s", v.endpoint, url.PathEscape(user.UserID), q.Encode())
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
	out := make([]string, 0, len(*result.Allowed))
	for _, id := range *result.Allowed {
		if name, ok := vaultCategoryNames[id]; ok {
			id = name
		}
		out = append(out, id)
	}
	return out, nil
}

func (v *httpClient) FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	return filterByCategorySet(results, categories)
}

// --- Mock ---

type mockClient struct{}

// NewMock returns a Vault Client for when Vault is disabled: there is no permission model, so
// AllowedCategories is nil (unrestricted) and nothing is filtered.
func NewMock() Client {
	return &mockClient{}
}

func (m *mockClient) AllowedCategories(_ context.Context, _ models.UserContext) ([]string, error) {
	return nil, nil
}

func (m *mockClient) FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	return filterByCategorySet(results, categories)
}

// filterByCategorySet keeps results in the given categories. nil means unrestricted; an empty,
// non-nil set means no access, so every result is removed.
func filterByCategorySet(results []models.SearchResult, categories []string) ([]models.SearchResult, int) {
	if categories == nil {
		return results, 0
	}
	allowed := make(map[string]struct{}, len(categories))
	for _, c := range categories {
		allowed[c] = struct{}{}
	}
	out := make([]models.SearchResult, 0, len(results))
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
