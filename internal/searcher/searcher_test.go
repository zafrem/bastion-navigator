package searcher

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func fakeQdrant(t *testing.T, search, info string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/points/search"):
			_, _ = w.Write([]byte(search))
		case r.URL.Path == "/collections":
			_, _ = w.Write([]byte(`{"result":{"collections":[{"name":"docs"}]}}`))
		default:
			_, _ = w.Write([]byte(info))
		}
	}))
	t.Cleanup(srv.Close)
	return strings.TrimPrefix(srv.URL, "http://")
}

func TestVectorSearchAcceptsIntegerIDsAndNonStringPayloads(t *testing.T) {
	host := fakeQdrant(t, `{"result":[
		{"id":42,"score":0.9,"payload":{"content":"x","category":"customer_data","page":3,"tags":["a"],"ok":true}},
		{"id":"7f2c0a1e-0000-0000-0000-000000000001","score":0.5,"payload":{"content":"y"}}]}`, "")
	got, err := NewQdrant([]string{host}).VectorSearch(context.Background(), "docs", []float32{1}, nil, 5, 0)
	if err != nil {
		t.Fatalf("a valid Qdrant response was rejected: %v", err)
	}
	if len(got) != 2 || got[0].DocumentID != "42" || got[1].DocumentID != "7f2c0a1e-0000-0000-0000-000000000001" {
		t.Fatalf("ids = %+v", got)
	}
	if got[0].Content != "x" || got[0].Category != "customer_data" || got[0].Metadata["page"] != "3" ||
		got[0].Metadata["tags"] != `["a"]` || got[0].Metadata["ok"] != "true" {
		t.Errorf("payload = %+v", got[0].Metadata)
	}
}

func TestCollectionCountPrefersPointsCountOverNullVectorsCount(t *testing.T) {
	host := fakeQdrant(t, "", `{"result":{"status":"green","vectors_count":null,"points_count":5}}`)
	cols, err := NewQdrant([]string{host}).Collections(context.Background())
	if err != nil || len(cols) != 1 || cols[0].VectorCount != 5 || cols[0].Status != "green" {
		t.Errorf("got %+v, %v; want 5 vectors and status green", cols, err)
	}
}

func TestCollectionCountFallsBackToVectorsCountOnOldQdrant(t *testing.T) {
	host := fakeQdrant(t, "", `{"result":{"status":"green","vectors_count":9}}`)
	cols, _ := NewQdrant([]string{host}).Collections(context.Background())
	if len(cols) != 1 || cols[0].VectorCount != 9 {
		t.Errorf("got %+v, want 9 from the legacy field", cols)
	}
}
