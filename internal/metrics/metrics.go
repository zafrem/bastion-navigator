package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Search operations
	SearchesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "navigator_searches_total",
		Help: "Total number of search requests.",
	}, []string{"tenant", "strategy"})

	SearchDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "navigator_search_duration_seconds",
		Help:    "Search request duration in seconds.",
		Buckets: []float64{0.01, 0.05, 0.1, 0.2, 0.5, 1.0},
	}, []string{"strategy"})

	ResultsReturned = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_results_returned_total",
		Help: "Total number of search results returned to callers.",
	})

	ResultsFiltered = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_results_filtered_total",
		Help: "Total number of candidates filtered out by permissions.",
	})

	ZeroResultSearches = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_zero_result_searches_total",
		Help: "Searches that returned zero results.",
	})

	// Embedding
	EmbeddingsGenerated = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_embeddings_generated_total",
		Help: "Total embeddings generated.",
	})

	EmbeddingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "navigator_embedding_duration_seconds",
		Help:    "Embedding generation duration.",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25},
	})

	// Reranking
	RerankingsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_rerankings_total",
		Help: "Total reranking operations.",
	})

	RerankDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "navigator_rerank_duration_seconds",
		Help:    "Reranking duration.",
		Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5},
	})

	// Cache
	CacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_cache_hits_total",
		Help: "Cache hits.",
	})

	CacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "navigator_cache_misses_total",
		Help: "Cache misses.",
	})

	// Vector DB
	QdrantCallsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "navigator_qdrant_calls_total",
		Help: "Total Qdrant API calls.",
	}, []string{"operation"})

	QdrantCallDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "navigator_qdrant_call_duration_seconds",
		Help:    "Qdrant call duration.",
		Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.5},
	})
)
