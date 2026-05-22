package vault

import (
	"testing"

	"github.com/bastion/navigator/internal/models"
)

func TestFilterByCategory_AllowsMatching(t *testing.T) {
	results := []models.SearchResult{
		{DocumentID: "1", Category: "customer_data"},
		{DocumentID: "2", Category: "hr_data"},
	}
	out, filtered := filterByCategory(results, []string{"customer_data"})
	if len(out) != 1 || out[0].DocumentID != "1" {
		t.Errorf("expected doc 1 only, got %v", out)
	}
	if filtered != 1 {
		t.Errorf("expected 1 filtered, got %d", filtered)
	}
}

func TestFilterByCategory_EmptyAllowed_PassesAll(t *testing.T) {
	results := []models.SearchResult{
		{DocumentID: "1", Category: "customer_data"},
		{DocumentID: "2", Category: "hr_data"},
	}
	out, filtered := filterByCategory(results, nil)
	if len(out) != 2 {
		t.Errorf("expected all 2 results, got %d", len(out))
	}
	if filtered != 0 {
		t.Errorf("expected 0 filtered, got %d", filtered)
	}
}

func TestFilterByCategory_UncategorisedPassesThrough(t *testing.T) {
	results := []models.SearchResult{
		{DocumentID: "1", Category: ""},
	}
	out, _ := filterByCategory(results, []string{"customer_data"})
	if len(out) != 1 {
		t.Errorf("uncategorised doc should pass through")
	}
}

func TestMockVault_AllCategories(t *testing.T) {
	m := NewMock()
	if len(m.AllCategories) == 0 {
		t.Error("MockVaultClient should have default allowed categories")
	}
}

func TestMockVault_FilterResults_RestrictsToAllowed(t *testing.T) {
	m := NewMock()
	results := []models.SearchResult{
		{DocumentID: "pub", Category: "customer_data"},
		{DocumentID: "priv", Category: "secret_data"},
	}
	out, filtered := m.FilterResults(results, []string{"customer_data"})
	if len(out) != 1 {
		t.Errorf("expected 1 result, got %d", len(out))
	}
	if filtered != 1 {
		t.Errorf("expected 1 filtered, got %d", filtered)
	}
}
