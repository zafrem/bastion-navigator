package embedder

import (
	"context"
	"math"
	"sync"
	"testing"
)

func norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func TestMockVectorsAreUnitLength(t *testing.T) {
	e := NewMock(128)
	v, err := e.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 128 {
		t.Fatalf("dim = %d, want 128", len(v))
	}
	if n := norm(v); math.Abs(n-1) > 1e-3 {
		t.Errorf("Embed norm = %v, want 1", n)
	}

	batch, err := e.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range batch {
		if n := norm(b); math.Abs(n-1) > 1e-3 {
			t.Errorf("EmbedBatch[%d] norm = %v, want 1", i, n)
		}
	}
}

// The mock backs a concurrent HTTP server; run with -race to catch unsynchronised use.
func TestMockIsSafeForConcurrentUse(t *testing.T) {
	e := NewMock(64)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				if _, err := e.Embed(context.Background(), "q"); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
