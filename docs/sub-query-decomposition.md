# Navigator — Sub-Query Decomposition

**Module:** Navigator (`navigator/decomposer.py`, `navigator/orchestrator.py`, `navigator/router.py`)
**Version:** 3.1
**Last updated:** 2026-05-31

---

## Overview

A multi-hop question — one whose answer requires chaining facts from multiple documents — retrieves poorly as a single query, because the embedding of the combined question sits "between" the topics and matches neither well. Sub-query decomposition (FR-MR-02-003) splits such a question into independent sub-queries, searches each in parallel, and merges the results with Reciprocal Rank Fusion.

Decomposition is **rule-based** — conjunction and clause splitting, no LLM call — so it adds negligible latency (NFR-MR-08 target: < 800 ms for N=4 parallel sub-searches, dominated by the searches themselves, not the split).

```
multi_hop query
    │
    ▼
QueryDecomposer.decompose()      ← rule-based split, ≤ 4 sub-queries
    │
    ▼
ThreadPoolExecutor               ← N sub-queries searched in parallel
    │
    ▼
RRF merge across all result sets
    │
    ▼
optional rerank → top_k
```

---

## Trigger: only on `multi_hop` intent

The router classifies intent first; decomposition fires only when the intent is `multi_hop`. For every other intent the query passes through unchanged:

```python
# navigator/orchestrator.py — inside search()

if (
    self._decomposer is not None
    and self._cfg.decomposer.enabled
    and self._router is not None
):
    routing = self._router.route(req.query, collections)
    if routing.intent == QueryIntent.MULTI_HOP:
        t_decomp = time.perf_counter()
        decomposed = self._decomposer.decompose(req.query, routing.intent.value)
        decomp_ms = (time.perf_counter() - t_decomp) * 1000
        if len(decomposed.sub_queries) > 1:
            if self._publisher:
                self._publisher.publish(event_sub_queries_decomposed(
                    tc, len(req.query), len(decomposed.sub_queries),
                    decomposed.strategy, decomp_ms,
                ))
            return self._search_decomposed(req, decomposed, collections, opts, filters, tc, start)
```

If the decomposer cannot split the query (no conjunction, no clause boundary), it returns a single-element list and the normal single-query path runs.

---

## The decomposer

### Three split strategies, tried in order

```python
# navigator/decomposer.py

class QueryDecomposer:
    def decompose(self, query: str, intent: str = "multi_hop") -> DecomposedQuery:
        if intent != "multi_hop" or not query.strip():
            return DecomposedQuery(sub_queries=[query], strategy="none", original=query)

        # 1. Conjunction split ("and also", "as well as", "그리고", "또한", ...)
        parts = self._split_conjunctions(query)
        if len(parts) > 1:
            return DecomposedQuery(_cap_and_clean(parts), "conjunction", query)

        # 2. Temporal/conditional clause split ("before", "after", "이전에", ...)
        parts = self._split_temporal(query)
        if len(parts) > 1:
            return DecomposedQuery(_cap_and_clean(parts), "temporal", query)

        # 3. Sentence-boundary split (last resort for long multi-sentence queries)
        parts = self._split_sentences(query)
        if len(parts) > 1:
            return DecomposedQuery(_cap_and_clean(parts), "sentence", query)

        return DecomposedQuery(sub_queries=[query], strategy="none", original=query)
```

### Bilingual conjunction patterns

English uses `\b` word boundaries; Korean conjunctions have no ASCII boundary so they are listed bare:

```python
# navigator/decomposer.py

_RE_EN = re.compile(
    r"\band\s+also\b|\bas\s+well\s+as\b|\bboth\b"
    r"|\bmoreover\b|\badditionally\b|\bfurthermore\b"
    r"|\balso\b(?=\s+(?:find|show|get|list|what|who|when|where))",
    re.IGNORECASE,
)
_RE_KR = re.compile(r"그리고|또한|및|더불어|아울러")

def _split_conjunctions(self, query: str) -> list[str]:
    combined = _RE_EN.pattern + "|" + _RE_KR.pattern
    parts = re.split(combined, query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]
```

The `\balso\b` alternative uses a lookahead so "also" only splits when it introduces a new imperative clause (`also find ...`), not when it is an adverb inside one clause ("I would also like the report").

### The N ≤ 4 cap (circuit breaker)

`_cap_and_clean` deduplicates and enforces the hard limit. The cap is a security control (SC-04) — it bounds the blast radius of an adversarial query engineered to explode into hundreds of sub-searches:

```python
# navigator/decomposer.py

MAX_SUB_QUERIES = 4

def _cap_and_clean(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) == MAX_SUB_QUERIES:
            break  # hard cap — decomposition explosion circuit breaker
    return out
```

---

## Parallel execution + RRF merge

`_search_decomposed` runs each sub-query as a full search (embedding + per-collection retrieval) on a thread pool, then folds the result sets together with RRF:

```python
# navigator/orchestrator.py

def _search_decomposed(self, req, decomposed, collections, opts, filters, tc, start) -> SearchResponse:
    max_workers = min(len(decomposed.sub_queries), self._cfg.decomposer.max_sub_queries)

    def _run_sub(sub_query: str) -> list[SearchResult]:
        vec = self._embedder.embed(
            self._rewriter.rewrite_text(sub_query) if self._rewriter else sub_query
        )
        results: list[SearchResult] = []
        for col in collections:
            results.extend(self._search_collection(
                col, sub_query, vec, filters, opts,
                opts.top_k * self._cfg.search_defaults.over_fetch_multiplier))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_sub, sq): sq for sq in decomposed.sub_queries}
        all_sets: list[list[SearchResult]] = []
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_sets.append(fut.result())
            except Exception:
                pass  # one failed sub-query does not sink the whole search

    # Fold result sets together with equal-weight RRF.
    if not all_sets:
        merged = []
    elif len(all_sets) == 1:
        merged = all_sets[0]
    else:
        merged = all_sets[0]
        for result_set in all_sets[1:]:
            merged = _rrf(merged, result_set)

    merged = _top_k(merged, opts.top_k)
    if opts.use_reranking and merged:
        merged = self._reranker.rerank(req.query, merged, opts.top_k)  # rerank against ORIGINAL query

    duration_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        request_id=req.request_id, results=merged,
        metadata=SearchMetadata(
            strategy="decompose+rrf",
            total_candidates=sum(len(s) for s in all_sets),
            filtered_out=0, final_count=len(merged),
        ),
        processing_time_ms=duration_ms,
    )
```

Two important details:

- **`as_completed` tolerates partial failure** — if one sub-query's collection is unavailable, the others still contribute. The search degrades rather than failing.
- **Rerank uses the original query**, not a sub-query. The cross-encoder scores each merged candidate against the user's actual multi-hop question, so the final ranking reflects relevance to the whole question, not just the fragment that retrieved it.

---

## Security constraint SC-04 (deferred re-entry)

The MR spec requires that **each sub-query be an independent security surface** — re-entering Sentinel-IN and Vault Phase-1 independently. The current implementation defers the per-sub-query Sentinel/Vault round-trip because:

1. Sub-queries are **machine-generated** by splitting the original query at fixed conjunction tokens — they are fragments of text that has **already cleared** Sentinel-IN and Vault Phase-1 upstream of Navigator.
2. The decomposition rule introduces no attacker-controlled novel content; it only partitions validated input.

What full SC-04 compliance requires is a `SentinelClient` and per-iteration `VaultPhase1Client` wired into the orchestrator. That is a service-topology decision (it creates a Navigator→Sentinel dependency), tracked as a follow-up rather than a code-complexity gap. The N ≤ 4 cap remains the active circuit breaker in the interim.

---

## Configuration

```yaml
decomposer:
  enabled: true
  max_sub_queries: 4   # hard cap; also bounds the thread pool size
```

---

## End-to-end trace

```
Query: "Compare the leave policy for employees hired before 2020
        and also the policy for contractors"

1. Router classifies intent = multi_hop  (conjunction "and also" + comparison)

2. QueryDecomposer.decompose():
     _split_conjunctions matches "and also"
     → ["Compare the leave policy for employees hired before 2020",
        "the policy for contractors"]
     strategy = "conjunction", 2 sub-queries

   emit event_sub_queries_decomposed(sub_query_count=2, strategy="conjunction")

3. _search_decomposed (ThreadPoolExecutor, max_workers=2):
     sub-query A → embed → search collections → 50 candidates
     sub-query B → embed → search collections → 50 candidates
     (run concurrently)

4. RRF merge:
     merged = _rrf(setA, setB)        # documents matching BOTH ranked highest
     merged = _top_k(merged, 10)

5. Rerank against the ORIGINAL question:
     reranker.rerank("Compare the leave policy ... and also ... contractors", merged, 10)

6. Response:
     metadata.strategy = "decompose+rrf"
     metadata.total_candidates = 100
     metadata.final_count = 10
```

---

## Related documents

- `navigator/docs/hybrid-reranking.md` — RRF fusion and cross-encoder reranking internals
- `navigator/docs/source-connectors.md` — enterprise data integration (MR-06)
- `docs/12_module_navigator_srs_v3.md` — full Navigator SRS (§11b)
- `docs/31_modular_rag_requirements.md` — FR-MR-02-003, SC-04 constraint
