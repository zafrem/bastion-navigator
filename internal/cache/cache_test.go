package cache

import (
	"context"
	"testing"
	"time"
)

func TestInMemoryCache_EmbeddingRoundTrip(t *testing.T) {
	c := NewInMemory()
	ctx := context.Background()
	text := "hello world"
	vec := []float32{0.1, 0.2, 0.3}

	if err := c.SetEmbedding(ctx, text, vec, time.Minute); err != nil {
		t.Fatal(err)
	}

	got, ok := c.GetEmbedding(ctx, text)
	if !ok {
		t.Fatal("expected cache hit")
	}
	if len(got) != len(vec) {
		t.Fatalf("expected %d dims, got %d", len(vec), len(got))
	}
	for i := range vec {
		if got[i] != vec[i] {
			t.Errorf("dim %d: expected %f, got %f", i, vec[i], got[i])
		}
	}
}

func TestInMemoryCache_PermissionsRoundTrip(t *testing.T) {
	c := NewInMemory()
	ctx := context.Background()
	cats := []string{"customer_data", "hr_data"}

	if err := c.SetPermissions(ctx, "user-1", cats, time.Minute); err != nil {
		t.Fatal(err)
	}
	got, ok := c.GetPermissions(ctx, "user-1")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if len(got) != len(cats) {
		t.Errorf("expected %d cats, got %d", len(cats), len(got))
	}
}

func TestInMemoryCache_Miss_ReturnsFalse(t *testing.T) {
	c := NewInMemory()
	_, ok := c.GetEmbedding(context.Background(), "not-cached")
	if ok {
		t.Error("expected cache miss")
	}
}

func TestInMemoryCache_Expiry(t *testing.T) {
	c := NewInMemory()
	ctx := context.Background()

	_ = c.SetEmbedding(ctx, "exp", []float32{1.0}, 1*time.Millisecond)
	time.Sleep(5 * time.Millisecond)

	_, ok := c.GetEmbedding(ctx, "exp")
	if ok {
		t.Error("expected expired entry to be a cache miss")
	}
}

func TestInMemoryCache_DifferentKeysAreSeparate(t *testing.T) {
	c := NewInMemory()
	ctx := context.Background()

	_ = c.SetEmbedding(ctx, "key-a", []float32{1.0}, time.Minute)
	_ = c.SetEmbedding(ctx, "key-b", []float32{2.0}, time.Minute)

	a, _ := c.GetEmbedding(ctx, "key-a")
	b, _ := c.GetEmbedding(ctx, "key-b")
	if a[0] != 1.0 || b[0] != 2.0 {
		t.Error("keys should be independent")
	}
}
