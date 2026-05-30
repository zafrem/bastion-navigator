# Navigator — Hybrid Search and Reranking

**Module:** Navigator (`navigator/searcher.py`, `navigator/reranker.py`, `navigator/orchestrator.py`)
**Version:** 3.0
**Last updated:** 2026-05-30

---

## Overview

Navigator supports three retrieval strategies that are selected per request, either by the adaptive router (MR-01) or by the caller via `SearchOptions.strategy`:

| Strategy | Description |
|---|---|
| `vector_only` | Cosine similarity over BGE-M3 embeddings |
| `hybrid` | RRF fusion of vector + BM25-proxy results |
| `hybrid+rerank` | RRF fusion followed by cross-encoder reranking |

The two composable components are:
- **Hybrid fusion** — combines dense vector search and sparse keyword search using Reciprocal Rank Fusion (RRF)
- **Reranking** — reorders the fused candidate list using a BGE cross-encoder that scores query–passage relevance directly

---

## Search flow

```
SearchRequest
    │
    ▼
_do_route()           → selects strategy and target collections
    │
    ▼
_search_collection()  → vector_only OR hybrid (RRF)
    │
    ▼
reranker.rerank()     → cross-encoder rerank (when use_reranking=True)
    │
    ▼
_top_k()              → score sort without reranker
    │
    ▼
_filter_by_purpose()  → purpose pre-filter
    │
    ▼
_merge_best()         → union across loop iterations (MR-03)
    │
    ▼
SearchResponse
```

Entry point: `orchestrator.search(req)` → `_search_collection(col, query, vec, filters, opts, over_fetch)`

---

## Over-fetch

Before reranking, Navigator fetches more candidates than the final `top_k` to give the reranker a wider field to re-order:

```python
# navigator/orchestrator.py

over_fetch = opts.top_k * self._cfg.search_defaults.over_fetch_multiplier
# Default: top_k=10, over_fetch_multiplier=5 → over_fetch=50
```

`over_fetch_multiplier` defaults to `5`. With `top_k=10`, the vector and sparse searches each retrieve up to 50 candidates before fusion. The reranker then selects the best 10 from those 50.

This matters because vector similarity and cross-encoder relevance are not the same signal. A document ranked 35th by cosine similarity may be the most relevant passage for a specific question once the cross-encoder evaluates it.

---

## Stage 1 — Vector search

```python
# navigator/searcher.py

def vector_search(
    self,
    collection: str,
    vector: list[float],
    filters: dict[str, str],
    top_k: int,
    min_score: float = 0.0,
) -> list[SearchResult]:
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    start = time.perf_counter()
    qdrant_filter = None
    if filters:
        # Each filter entry becomes a Qdrant FieldCondition with match.value.
        # All conditions are joined by "must" (logical AND).
        # tenant_id is always in filters — see orchestrator._search_collection.
        qdrant_filter = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v))
                  for k, v in filters.items()]
        )

    hits = self._client.search(
        collection_name=collection,
        query_vector=vector,              # BGE-M3 1024-dim embedding
        query_filter=qdrant_filter,
        limit=top_k,
        score_threshold=min_score if min_score > 0 else None,
    )
    metrics.qdrant_call_duration_seconds.labels(operation="vector_search").observe(
        time.perf_counter() - start
    )
    return [_to_search_result(h) for h in hits]
```

Qdrant returns cosine similarity scores in `[−1, 1]` (stored as `[0, 1]` after normalization). Each hit is converted to a `SearchResult` via `_to_search_result()`.

### `_to_search_result()` — payload extraction

```python
# navigator/searcher.py

# Provenance fields are extracted as first-class attributes.
# Every other payload key becomes a metadata entry.
_PROVENANCE_KEYS = frozenset({
    "content", "document_id", "chunk_id", "heading_path",
    "char_start", "char_end", "last_indexed",
})

def _to_search_result(hit) -> SearchResult:
    payload = hit.payload or {}
    return SearchResult(
        document_id=payload.get("document_id", str(hit.id)),
        content=payload.get("content", ""),
        score=hit.score,                                 # cosine similarity from Qdrant
        chunk_id=payload.get("chunk_id", ""),
        heading_path=payload.get("heading_path", ""),
        char_start=int(payload["char_start"]) if payload.get("char_start") is not None else 0,
        char_end=int(payload["char_end"])   if payload.get("char_end")   is not None else 0,
        last_indexed=payload.get("last_indexed", ""),
        # All non-provenance keys land in metadata (tenant_id, category, etc.)
        metadata={k: v for k, v in payload.items() if k not in _PROVENANCE_KEYS},
    )
```

---

## Stage 2 — Sparse search (BM25-proxy)

```python
# navigator/searcher.py

def sparse_search(
    self,
    collection: str,
    query: str,
    filters: dict[str, str],
    top_k: int,
) -> list[SearchResult]:
    # Uses Qdrant's scroll() to retrieve filtered chunks without vector ranking.
    hits = self._client.scroll(
        collection_name=collection,
        scroll_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )[0]

    results = [_to_search_result(h) for h in hits]

    # BM25-proxy scoring: count query token overlaps with chunk content.
    # score = matched_token_count / total_query_tokens
    q_lower = query.lower()
    for r in results:
        r.score = sum(
            1 for w in q_lower.split() if w in r.content.lower()
        ) / max(len(q_lower.split()), 1)

    return sorted(results, key=lambda r: r.score, reverse=True)
```

The sparse scorer is a token-overlap proxy for BM25. It normalises the match count by query length, so a 2-word query that matches both words scores `1.0`, while a 10-word query that matches 3 words scores `0.3`. This is not term-frequency weighted like true BM25 but provides keyword signal for RRF at low latency.

---

## Stage 3 — Reciprocal Rank Fusion (RRF)

RRF merges the vector and sparse ranked lists into a single fused ranking. It is rank-based, not score-based, so the absolute similarity values from the two systems never need to be on the same scale.

```python
# navigator/orchestrator.py

def _rrf(
    vector: list[SearchResult],
    bm25: list[SearchResult],
    vw: float = 0.7,   # vector weight
    bw: float = 0.3,   # BM25 weight
    k: float = 60.0,   # smoothing constant (standard RRF default)
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of two ranked lists."""
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}

    # Vector contribution: vw / (k + rank + 1)
    # rank is 0-indexed over the list sorted by descending score.
    for rank, r in enumerate(sorted(vector, key=lambda r: r.score, reverse=True)):
        scores[r.document_id] = scores.get(r.document_id, 0) + vw / (k + rank + 1)
        docs[r.document_id] = r

    # BM25 contribution: bw / (k + rank + 1)
    for rank, r in enumerate(sorted(bm25, key=lambda r: r.score, reverse=True)):
        scores[r.document_id] = scores.get(r.document_id, 0) + bw / (k + rank + 1)
        docs[r.document_id] = r

    # Sort by fused RRF score descending.
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[doc_id] for doc_id, _ in ranked]
```

### RRF score mechanics

For a document ranked position `r` (0-indexed) in a list:

```
contribution = weight / (60 + r + 1)
```

| Rank (r) | Vector (vw=0.7) | BM25 (bw=0.3) |
|---|---|---|
| 0 (top) | 0.7 / 61 = 0.01148 | 0.3 / 61 = 0.00492 |
| 4 | 0.7 / 65 = 0.01077 | 0.3 / 65 = 0.00462 |
| 49 (bottom) | 0.7 / 110 = 0.00636 | 0.3 / 110 = 0.00273 |

The `k=60` constant compresses the score difference between top and bottom ranks, preventing any single high-ranked result from dominating.

**Cross-list bonus:** A document that appears in both the vector list (rank 2) and the BM25 list (rank 5) accumulates both contributions:
```
score = 0.7/(60+3) + 0.3/(60+6) = 0.01111 + 0.00455 = 0.01566
```
This is why hybrid search outperforms either modality alone: documents that are both semantically close AND keyword-matched get promoted above documents that only satisfy one criterion.

### `_search_collection()` — the decision gate

```python
# navigator/orchestrator.py

def _search_collection(
    self,
    collection: str,
    query: str,
    vec: list[float],
    filters: dict[str, str],
    opts: SearchOptions,
    over_fetch: int,
) -> list[SearchResult]:
    # Always perform vector search.
    vector_results = self._searcher.vector_search(collection, vec, filters, over_fetch)

    if not opts.use_hybrid:
        return vector_results  # vector_only path

    # Hybrid path: run sparse search and fuse with RRF.
    sparse_results = self._searcher.sparse_search(collection, query, filters, over_fetch)
    return _rrf(vector_results, sparse_results)
    # Note: vw=0.7, bw=0.3, k=60 are hardcoded in _rrf.
    # opts.vector_weight and opts.bm25_weight are stored on SearchOptions
    # for future per-request tuning but not currently wired into _rrf.
```

---

## Stage 4 — Cross-encoder reranking

Reranking takes the fused candidate list and scores each (query, passage) pair directly using a cross-encoder — a model that reads both texts together rather than comparing independent embeddings.

### `Reranker` interface

```python
# navigator/reranker.py

class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]: ...
```

### `LocalReranker` — in-process BGE cross-encoder

```python
class LocalReranker(Reranker):
    """Runs the BGE-reranker-v2-m3 cross-encoder in-process."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder
        # Model loaded once at startup; shared across all requests.
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        start = time.perf_counter()

        # Build (query, passage) pairs for the cross-encoder.
        # The cross-encoder reads both texts in a single forward pass,
        # allowing it to see query-passage interaction that bi-encoder
        # embeddings (like BGE-M3) cannot capture.
        pairs = [[query, c.content] for c in candidates]

        # predict() returns a float score per pair; higher = more relevant.
        scores = self._model.predict(pairs)

        metrics.rerank_duration_seconds.observe(time.perf_counter() - start)

        # Sort by reranker score descending; keep top_k.
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
```

### `BGEHttpReranker` — remote service

```python
class BGEHttpReranker(Reranker):
    """Calls a remote reranker service over HTTP."""

    def __init__(self, endpoint: str, timeout: float = 10.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        start = time.perf_counter()
        resp = httpx.post(
            f"{self._endpoint}/rerank",
            json={
                "query": query,
                "candidates": [c.model_dump() for c in candidates],
                "top_k": top_k,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        metrics.rerank_duration_seconds.observe(time.perf_counter() - start)

        # Re-map remote results back to local SearchResult objects by document_id.
        # The remote service returns {document_id, ...} entries in ranked order.
        ids = {c.document_id: c for c in candidates}
        return [ids[r["document_id"]] for r in resp.json()["results"]
                if r["document_id"] in ids]
```

### Reranker selection

```python
# navigator/reranker.py

def build(cfg: RerankerConfig) -> Reranker:
    if cfg.type == "local":
        return LocalReranker(cfg.model_name, cfg.max_length)
    elif cfg.type == "bge_http":
        return BGEHttpReranker(cfg.endpoint)
    return MockReranker()  # test/dev
```

Config path (`config.yaml`):
```yaml
reranker:
  enabled: true
  type: local                         # "local" | "bge_http"
  model_name: BAAI/bge-reranker-v2-m3
  max_length: 512
```

---

## Strategy selection — router to `SearchOptions`

The adaptive router classifies the query intent and emits a strategy string. `_apply_routing_strategy()` translates that string into `use_hybrid` and `use_reranking` flags on `SearchOptions`:

```python
# navigator/orchestrator.py

def _apply_routing_strategy(opts: SearchOptions, strategy: str) -> SearchOptions:
    if strategy == "vector_only":
        return opts.model_copy(update={"use_hybrid": False, "use_reranking": False})
    if strategy == "hybrid":
        return opts.model_copy(update={"use_hybrid": True, "use_reranking": False})
    if strategy == "hybrid+rerank":
        return opts.model_copy(update={"use_hybrid": True, "use_reranking": True})
    return opts  # unrecognised strategy — preserve caller's values
```

Intent → strategy mapping (defined in `router.py`):

```python
_INTENT_STRATEGY: dict[QueryIntent, str] = {
    QueryIntent.FACTUAL:    "vector_only",    # short lookup — vector is sufficient
    QueryIntent.ANALYTICAL: "hybrid+rerank",  # needs keyword + semantic + ordering
    QueryIntent.PROCEDURAL: "hybrid",         # keyword terms important; rerank costly
    QueryIntent.MULTI_HOP:  "hybrid+rerank",  # multi-entity — full pipeline
    QueryIntent.AMBIGUOUS:  "hybrid",         # no clear intent — safe default
}
```

The caller can bypass routing by setting `SearchOptions.strategy` directly:

```python
SearchOptions(strategy="hybrid+rerank")  # forces full pipeline regardless of intent
```

### Strategy name in metrics

```python
# navigator/orchestrator.py

def _strategy_name(opts: SearchOptions) -> str:
    if opts.use_hybrid and opts.use_reranking:
        return "hybrid+rerank"
    if opts.use_hybrid:
        return "hybrid"
    if opts.use_reranking:
        return "vector+rerank"
    return "vector"
```

This string is stored in `SearchMetadata.strategy` and emitted as a Prometheus label:
```python
metrics.searches_total.labels(strategy=strategy, tenant_id=req.tenant_id).inc()
metrics.search_duration_seconds.labels(strategy=strategy).observe(duration_ms / 1000)
```

---

## SearchOptions and SearchResult score fields

```python
# navigator/models.py

class SearchOptions(BaseModel):
    top_k: int = 0
    over_fetch: int = 0
    use_reranking: bool = False
    use_hybrid: bool = False
    vector_weight: float = 0.0       # stored; not yet wired into _rrf
    bm25_weight: float = 0.0         # stored; not yet wired into _rrf
    filters: dict[str, str] = {}
    min_score: float = 0.0
    strategy: str = ""               # override router (MR-01)
    use_hyde: bool = False           # HyDE embedding override (MR-02-002)

class SearchResult(BaseModel):
    document_id: str = ""
    content: str = ""
    score: float = 0.0           # final score (RRF or reranker)
    vector_score: float = 0.0    # raw cosine from Qdrant (populated by future work)
    bm25_score: float = 0.0      # raw keyword score (populated by future work)
    rerank_score: float = 0.0    # cross-encoder score (populated by future work)
    # ...provenance fields
```

`score` is the value used for all sorting decisions. The `vector_score`, `bm25_score`, and `rerank_score` fields are reserved for per-signal observability but are not currently populated by the retrieval path.

---

## Configuration reference

```yaml
search_defaults:
  top_k: 10
  over_fetch_multiplier: 5    # candidate pool = top_k × multiplier
  use_hybrid: true            # default-on for all requests
  use_reranking: true         # default-on for all requests
  vector_weight: 0.7          # future per-request RRF weight
  bm25_weight: 0.3
  min_score: 0.5              # drop results below this score after reranking
  timeout_ms: 500

reranker:
  enabled: true
  type: local                 # "local" | "bge_http"
  model_name: BAAI/bge-reranker-v2-m3
  endpoint: http://localhost:8001
  max_length: 512
```

---

## End-to-end trace — `hybrid+rerank`

**Query:** `"Compare defect rates for line A and line B in Q1"`

**Router decision:**
```
_RE_ANALYTICAL matches "compare" → scores[ANALYTICAL] += 0.6
_RE_MULTI_HOP  matches "and"     → scores[MULTI_HOP]  += 0.7

fired = 2 → best_score *= 0.7 → MULTI_HOP: 0.7*0.7 = 0.49

intent    = MULTI_HOP
strategy  = "hybrid+rerank"
_apply_routing_strategy → use_hybrid=True, use_reranking=True
```

**`_search_collection("manufacturing_docs", query, vec, filters, opts, over_fetch=50)`:**
```
vector_search("manufacturing_docs", vec, {"tenant_id":"acme-corp"}, 50)
  → 50 hits ranked by cosine similarity (BGE-M3 embedding)

sparse_search("manufacturing_docs", "compare defect rates ...", {"tenant_id":"acme-corp"}, 50)
  → 50 hits ranked by token-overlap score
    "defect" in content → +1; "line" → +1; "compare" → +1; ...
    score = matched/total_query_tokens

_rrf(vector_results, sparse_results, vw=0.7, bw=0.3, k=60)
  → fused list of up to 50 unique docs
  → docs appearing in both lists promoted
```

**`reranker.rerank(query, fused_50, top_k=10)`:**
```
LocalReranker:
  pairs = [
    ["Compare defect rates ...", "Line A had 12 defects in Q1..."],
    ["Compare defect rates ...", "Q1 production summary for all lines..."],
    ...  ×50
  ]
  scores = CrossEncoder.predict(pairs)  → [0.91, 0.87, 0.62, 0.45, ...]
  → sorted descending → return top 10
```

**Result:**
```json
{
  "results": [
    {"document_id": "mfg_report_q1", "score": 0.91, "content": "Line A had 12 defects in Q1..."},
    {"document_id": "mfg_summary",   "score": 0.87, "content": "Q1 production summary..."},
    ...
  ],
  "metadata": {
    "strategy": "hybrid+rerank",
    "total_candidates": 50,
    "final_count": 10
  }
}
```

---

## Related documents

- `navigator/docs/CONFIGURATION.md` — full config schema
- `navigator/docs/logical-partitioning.md` — collection routing and tenant filtering
- `docs/12_module_navigator_srs_v3.md` — full Navigator SRS
- `docs/31_modular_rag_requirements.md` — MR-01 routing, MR-03 re-search loop
