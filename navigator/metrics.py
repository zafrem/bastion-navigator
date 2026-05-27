"""Prometheus metrics for Navigator."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

searches_total = Counter(
    "navigator_searches_total",
    "Total search requests",
    ["strategy", "tenant_id"],
)
search_duration_seconds = Histogram(
    "navigator_search_duration_seconds",
    "End-to-end search latency",
    ["strategy"],
)
results_returned = Histogram(
    "navigator_results_returned",
    "Number of results returned per search",
    ["collection"],
)
results_filtered = Counter(
    "navigator_results_filtered_total",
    "Results removed by permission or tenant filter",
)
zero_result_searches = Counter(
    "navigator_zero_result_searches_total",
    "Searches that returned zero results",
)
embedding_duration_seconds = Histogram(
    "navigator_embedding_duration_seconds",
    "Time to embed a single query",
)
rerank_duration_seconds = Histogram(
    "navigator_rerank_duration_seconds",
    "Time to rerank candidates",
)
qdrant_call_duration_seconds = Histogram(
    "navigator_qdrant_call_duration_seconds",
    "Time spent in Qdrant calls",
    ["operation"],
)
cache_hits = Counter("navigator_cache_hits_total", "Embedding cache hits")
cache_misses = Counter("navigator_cache_misses_total", "Embedding cache misses")
