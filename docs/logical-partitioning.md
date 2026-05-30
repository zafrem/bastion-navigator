# Navigator — Logical Partitioning

**Module:** Navigator (`navigator/chunker.py`, `navigator/searcher.py`, `navigator/orchestrator.py`, `navigator/router.py`)
**Version:** 3.0
**Last updated:** 2026-05-30

---

## Overview

Logical partitioning in Navigator has two orthogonal axes:

| Axis | Mechanism | Enforced at |
|---|---|---|
| **Tenant isolation** | `tenant_id` filter on every Qdrant query | Search time |
| **Data category routing** | Separate Qdrant collections per data category | Index time + search time |

These two axes are independent: a tenant's customer documents are stored in `customer_docs`, and the `tenant_id` payload field ensures no tenant can retrieve another tenant's chunks from that collection.

Additionally, within a document, the chunker partitions content by structural units (headings, tables, code fences) and attaches a heading breadcrumb to each chunk so the embedding model receives full structural context.

---

## Axis 1 — Data category collections

### Collection mapping

```python
# navigator/orchestrator.py

# Maps data category label (from IndexRequest.category / Vault RBAC)
# to the Qdrant collection name that stores those documents.
_CATEGORY_TO_COLLECTION: dict[str, str] = {
    "customer_data":      "customer_docs",
    "manufacturing_data": "manufacturing_docs",
    "hr_data":            "hr_docs",
}
```

### Index time — writing to the correct collection

```python
# navigator/orchestrator.py

def index_document(self, req: IndexRequest) -> IndexResponse:
    # ...chunk and embed...

    # The target collection is derived from IndexRequest.category.
    # If the caller omits category, documents land in "default".
    collection = req.category or "default"

    # Qdrant creates the collection on first upsert if it does not exist.
    self._searcher.ensure_collection(collection, vector_size=len(vectors[0]))

    points = []
    for chunk, vec in zip(chunks, vectors):
        points.append({
            "id": chunk.stable_uuid(),     # deterministic UUID from chunk_id
            "vector": vec,
            "payload": {
                "document_id":       req.document_id,
                "chunk_id":          chunk.chunk_id,
                "tenant_id":         req.tenant_id,
                "category":          req.category,   # also stored in payload
                "heading_path":      " > ".join(chunk.heading_path),
                "char_start":        chunk.char_start,
                "char_end":          chunk.char_end,
                "last_indexed":      last_indexed,
                "permitted_purposes": ",".join(req.permitted_purposes),
                "content":           chunk.content,
                **req.metadata,
            },
        })
    self._searcher.upsert(collection, points)
```

`IndexRequest.category` carries the data category declared by the Vault client. The orchestrator maps it to a collection name and ensures the collection exists before upserting.

### Search time — selecting target collections

```python
# navigator/orchestrator.py

# Step 1: resolve the user's allowed data categories from Vault RBAC.
allowed = self._resolve_permissions(req)
# e.g. ["customer_data", "manufacturing_data"]

# Step 2: translate allowed categories to collection names.
permission_collections = self._collections_for_categories(allowed)
# e.g. ["customer_docs", "manufacturing_docs"]

# Step 3: MR-01 router may further narrow to collections relevant to the query.
collections, opts = self._do_route(req.query, permission_collections, opts, tc)
# e.g. ["manufacturing_docs"]
```

### `_resolve_permissions()` — RBAC-sourced categories

```python
# navigator/orchestrator.py

def _resolve_permissions(self, req: SearchRequest) -> Optional[list[str]]:
    # Fast path: allowed_categories already carried in UserContext.
    if req.user and req.user.allowed_categories:
        return req.user.allowed_categories

    # Slow path: ask Vault for categories permitted for this user.
    user_id = req.user.user_id if req.user else ""
    if not user_id:
        return None   # no user → fall through to "all collections" default
    return self._vault.allowed_categories(user_id)
```

### `_collections_for_categories()` — category → collection translation

```python
def _collections_for_categories(self, allowed: Optional[list[str]]) -> list[str]:
    if not allowed:
        # No permission restriction: search all known collections.
        # Falls back to hardcoded _CATEGORY_TO_COLLECTION values if
        # config.vector_db.collections is empty.
        return (
            list(self._cfg.vector_db.collections.keys())
            or list(_CATEGORY_TO_COLLECTION.values())
        )

    seen: set[str] = set()
    out: list[str] = []
    for cat in allowed:
        col = _CATEGORY_TO_COLLECTION.get(cat)
        if col and col not in seen:
            seen.add(col)
            out.append(col)

    # If no mapping found (unknown category), default to all collections.
    return out or list(_CATEGORY_TO_COLLECTION.values())
```

---

## Axis 2 — Tenant isolation within a collection

Every chunk stored in Qdrant carries a `tenant_id` payload field. Every search query includes a `tenant_id` filter that Qdrant evaluates as a pre-filter before ANN search.

### Filter construction

```python
# navigator/orchestrator.py

def search(self, req: SearchRequest, ...) -> SearchResponse:
    # tenant_id is always the first filter entry.
    # Additional caller-supplied filters are merged on top.
    filters: dict[str, str] = {"tenant_id": req.tenant_id}
    if opts.filters:
        filters.update(opts.filters)
    # ...
    # filters is passed to every _search_collection() call.
```

### Filter application in Qdrant

```python
# navigator/searcher.py

def _build_filter(filters: dict[str, str]) -> dict:
    """Convert a flat key=value dict into a Qdrant filter."""
    conditions = [
        {"key": k, "match": {"value": v}}
        for k, v in filters.items()
    ]
    return {"must": conditions} if conditions else {}

def vector_search(self, collection, vector, filters, top_k, min_score=0.0):
    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(
            # ALL conditions must match (logical AND).
            # {"tenant_id": "acme-corp"} → only chunks owned by "acme-corp"
            must=[FieldCondition(key=k, match=MatchValue(value=v))
                  for k, v in filters.items()]
        )
    hits = self._client.search(
        collection_name=collection,
        query_vector=vector,
        query_filter=qdrant_filter,   # applied before ANN retrieval
        limit=top_k,
    )
    # ...
```

The same `qdrant_filter` is applied in `sparse_search()` (the `scroll_filter` argument), so tenant isolation is enforced on both retrieval paths before their results are fused.

### Why collection-level isolation is not sufficient

Two tenants' data can coexist in the same collection. The collection boundary alone does not prevent cross-tenant reads — the `tenant_id` filter does. The collection is a performance partition (each collection has its own HNSW graph); the tenant filter is the security partition.

---

## Axis 3 — Router-based domain narrowing (MR-01)

Within the set of permission-allowed collections, the router can narrow further by matching the query against per-collection domain keywords:

```python
# navigator/router.py

# Domain keywords per collection — lightweight topic proxies.
# English and Korean terms are mixed in the same list.
_COLLECTION_DOMAINS: dict[str, list[str]] = {
    "customer_docs":      ["customer", "account", "purchase", "고객", "계좌", "구매", "주문"],
    "manufacturing_docs": ["defect", "production", "factory", "line", "worker", "불량", "생산", "공장", "공정"],
    "hr_docs":            ["employee", "salary", "leave", "hr", "직원", "급여", "연차", "인사", "휴가", "근태"],
}
```

### `_select_collections()` — keyword hit counting

```python
def _select_collections(
    self,
    query: str,
    available: list[str],
) -> tuple[list[str], list[str]]:
    q_lower = query.lower()
    hits: dict[str, int] = {}
    for col in available:
        keywords = _COLLECTION_DOMAINS.get(col, [])
        # Count how many domain keywords appear as substrings of the lowercased query.
        hits[col] = sum(1 for kw in keywords if kw in q_lower)

    if max(hits.values(), default=0) == 0:
        # No domain signal at all → search all available collections (fail-open).
        # This follows SC-03 (conservative default) — never silently drop results
        # when the intent is ambiguous.
        return list(available), []

    selected = [c for c, h in hits.items() if h > 0]
    excluded = [c for c, h in hits.items() if h == 0]
    return selected, excluded
```

A keyword hit count of `0` for all collections causes a fail-open: all collections are searched. A hit in any collection causes only the matching collections to be searched, reducing Qdrant round-trips.

### Routing in `_do_route()`

```python
# navigator/orchestrator.py

def _do_route(self, query, permission_collections, opts, tc):
    if self._router is None:
        return permission_collections, opts  # routing disabled

    try:
        routing = self._router.route(
            query,
            permission_collections,       # router only selects within permitted set
            routing_threshold=self._cfg.modular_rag.router.routing_threshold,
            strategy_override=opts.strategy,
        )
        if self._publisher:
            self._publisher.publish(event_query_routed(
                tc, routing.intent.value, routing.strategy,
                routing.collections, routing.excluded,
                routing.confidence, routing.routing_ms,
            ))
        updated_opts = _apply_routing_strategy(opts, routing.strategy)
        # Fall back to full permission_collections if router returns empty list.
        return routing.collections or permission_collections, updated_opts

    except Exception:
        # Router failure is non-fatal: fall back to all permitted collections.
        return permission_collections, opts
```

The router always receives `permission_collections` as its available set, never the full collection list. It can narrow further but never expand beyond what RBAC permits.

---

## Document-level partitioning — the chunker

### Why chunking matters for partitioning

Vector databases store and retrieve at chunk granularity, not document granularity. A 10,000-character document is split into ~8–10 chunks. Each chunk is an independent Qdrant point with its own vector, `tenant_id`, and payload. Retrieval returns individual chunks, not full documents.

The chunker's job is to produce chunks that:
- Fit within the embedding model's context window (BGE-M3 max 512 tokens ≈ 2,000 chars)
- Carry enough structural context to be meaningful in isolation (heading breadcrumb)
- Do not split semantic units (tables, code fences) mid-structure

### `ChunkerConfig`

```python
# navigator/chunker.py

@dataclass
class ChunkerConfig:
    max_chars: int = 1200       # ~300 tokens at 4 chars/token
    overlap_chars: int = 120    # tail of previous chunk prepended to next
    min_chars: int = 80         # chunks smaller than this are merged into previous
```

### `Chunk` — the unit of storage

```python
@dataclass
class Chunk:
    chunk_id: str                 # "{parent_document_id}_{index:04d}"
    parent_document_id: str
    chunk_index: int
    content: str
    heading_path: list[str] = field(default_factory=list)
    contains_table: bool = False
    contains_link: bool = False
    char_start: int = 0           # byte offset in original document
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """Returns text sent to the embedder: heading breadcrumb + content.

        The breadcrumb prepended here is the core of structural partitioning:
        a chunk under "## Security > ### Injection Defense" embeds with
        that context, so vector search can distinguish it from a chunk with
        identical content under a different section.
        """
        if self.heading_path:
            breadcrumb = " > ".join(self.heading_path)
            return f"{breadcrumb}\n\n{self.content}"
        return self.content

    def stable_uuid(self) -> str:
        """Deterministic UUID derived from chunk_id using uuid5.

        Determinism means re-indexing a document overwrites existing Qdrant
        points for the same chunks (same ID = upsert, not insert).
        Non-determinism would cause orphaned points to accumulate on re-index.
        """
        return str(uuid.uuid5(_UUID_NS, self.chunk_id))
```

### Block parser — `_parse_blocks()`

The document is first split into typed blocks before being packed into chunks. This ensures structural units are never broken across chunks:

```python
# navigator/chunker.py

def _parse_blocks(text: str) -> list[_Block]:
    """Split document text into typed blocks: headings, tables, code fences, paragraphs."""
    # ...line-by-line iteration...

    # Heading: single line matching ^#{1,6}\s+(.*)
    # → _Block(is_heading=True, heading_level=N)

    # Code fence: starts with ```, collects until closing ```
    # → _Block(text=entire_fence)  — never split

    # Table: consecutive lines starting with |
    # → _Block(is_table=True, text=all_rows)  — never split

    # Paragraph: consecutive non-blank, non-heading, non-table, non-fence lines
    # → _Block(text=paragraph_text)
```

**Key invariant:** tables and code fences are collected atomically into a single `_Block`. They are never passed to `_split_oversized()` and are stored as self-contained chunks even if they exceed `max_chars`.

### Heading hierarchy maintenance

```python
# navigator/chunker.py

def chunk_document(document_id, content, cfg, metadata) -> list[Chunk]:
    heading_path: list[str] = []  # mutable stack of ancestor headings
    # ...

    for block in blocks:
        if block.is_heading:
            _flush(block.char_start)  # emit the accumulated buffer as a chunk

            level = block.heading_level  # 1–6
            # Truncate the stack to the parent level:
            # H3 encountered after H1 > H2 pops to depth 2 first.
            heading_path = heading_path[:level - 1]
            heading_path.append(block.text)
            # Heading text itself is NOT emitted as a chunk;
            # it becomes context for the next chunk via embed_text().
            overlap = ""   # heading boundary resets overlap
            continue

        if block.is_table:
            _flush(block.char_start)  # flush text buffer before table
            # Table becomes its own chunk immediately (atomic)
            chunks.append(Chunk(
                ...,
                heading_path=list(heading_path),  # snapshot of current path
                contains_table=True,
                ...
            ))
            continue

        _append_text_block(block)  # accumulate paragraph into buffer
```

Heading updates flush the current buffer first, so no chunk spans a heading boundary. Every chunk's `heading_path` is a snapshot of the ancestor hierarchy at the point of chunk creation.

### Oversized block splitting — `_split_oversized()`

```python
def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a block exceeding max_chars at sentence boundaries using NLTK Punkt.

    Punkt is used because:
    - It handles abbreviations: "Dr. Kim" does not split after "Dr."
    - It handles Korean sentence endings (없습니다, 합니다) that punctuation-only
      splitting would miss.
    - Falls back to hard char-cut with overlap if punkt data is unavailable.
    """
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = None

    if sentences:
        return _pack_sentences(sentences, max_chars, overlap_chars, text)

    # Hard-cut fallback: slice at max_chars, carry overlap_chars into next part.
    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        parts.append(remaining[:max_chars].strip())
        remaining = remaining[max(0, max_chars - overlap_chars):].lstrip()
    if remaining.strip():
        parts.append(remaining.strip())
    return parts or [text]
```

### Sentence packing with overlap — `_pack_sentences()`

```python
def _pack_sentences(sentences, max_chars, overlap_chars, original) -> list[str]:
    """Pack sentences into parts ≤ max_chars, carrying the tail of each part
    into the next as context overlap.

    Example with max_chars=100, overlap_chars=20:
      sentences = ["Alice works at ACME.", "She is the manager.", "Bob reports to her."]
      Part 1: "Alice works at ACME. She is the manager."  (78 chars, fits)
      Next sentence "Bob reports to her." would push to 98 chars — still fits.
      → Part 1: all three sentences. (This example fits; a longer example would split.)

    When a sentence pushes the buffer over max_chars:
      tail_of_part_1[-overlap_chars:] is prepended to part_2
      → part_2 = "o her.\n\nNext sentence..."
      This carries the end of part 1 into part 2, preserving cross-boundary context.
    """
    parts: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = (buf + " " + sent).lstrip() if buf else sent
        if buf and len(candidate) > max_chars:
            parts.append(buf.strip())
            # Carry the last overlap_chars of buf into the next part.
            overlap = buf[-overlap_chars:].lstrip() if overlap_chars else ""
            buf = (overlap + " " + sent).lstrip() if overlap else sent
        else:
            buf = candidate
    if buf.strip():
        parts.append(buf.strip())
    return parts or [original]
```

### Orphan tail merge

```python
# navigator/chunker.py

# After all blocks are processed, check if the last chunk is too small.
# If so, merge it into its predecessor — but only when they share the same
# heading section. Merging across section boundaries would corrupt the
# heading_path of the merged chunk.
if (
    len(chunks) >= 2
    and len(chunks[-1].content) < cfg.min_chars
    and chunks[-1].heading_path == chunks[-2].heading_path  # same section
):
    tail = chunks.pop()
    prev = chunks[-1]
    merged = prev.content + "\n\n" + tail.content
    chunks[-1] = Chunk(
        chunk_id=prev.chunk_id,   # keep predecessor's ID
        ...,
        content=merged,
        contains_table=prev.contains_table or tail.contains_table,
        char_start=prev.char_start,
        char_end=tail.char_end,   # extends to end of tail
    )
```

---

## Qdrant collection configuration

Each collection in `config.yaml` can have its own HNSW parameters, allowing recall/latency trade-offs to be tuned per data category:

```yaml
# config/config.yaml

vector_db:
  type: qdrant
  hosts: ["localhost:6333"]
  collections:
    customer_docs:
      vector_size: 1024      # BGE-M3 output dimension
      distance: cosine
      hnsw:
        m: 16                # number of bi-directional edges per node
        ef_construct: 200    # candidates considered during index construction
        ef_search: 128       # candidates considered during query
    manufacturing_docs:
      vector_size: 1024
      distance: cosine
      hnsw:
        m: 16
        ef_construct: 200
        ef_search: 128
    hr_docs:
      vector_size: 1024
      distance: cosine
      hnsw:
        m: 16
        ef_construct: 200
        ef_search: 128
```

```python
# navigator/config.py

class HNSWConfig(BaseModel):
    m: int = 16              # graph connectivity; higher = better recall, more memory
    ef_construct: int = 200  # build-time candidate set; higher = better index quality
    ef_search: int = 128     # query-time candidate set; higher = better recall, slower

class CollectionConfig(BaseModel):
    vector_size: int = 1024
    distance: str = "cosine"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
```

`ef_search` is the key latency/recall knob: increasing it improves recall at the cost of query latency. Different collections can use different values — a high-value HR collection might use `ef_search: 256` while a large customer collection uses `ef_search: 64`.

---

## End-to-end trace — index + search

### Indexing a manufacturing document

```
IndexRequest:
  document_id  = "mfg_defect_report_2026q1"
  tenant_id    = "acme-corp"
  category     = "manufacturing_data"
  content      = "# Q1 Defect Report\n\n## Line A\n\nLine A produced 450 units..."
  permitted_purposes = ["customer_support", "audit"]
```

**`chunk_document()` output (abbreviated):**
```
_parse_blocks() →
  _Block(is_heading, level=1, "# Q1 Defect Report")
  _Block(is_heading, level=2, "## Line A")
  _Block("Line A produced 450 units...")

chunk_document() →
  Chunk(
    chunk_id     = "mfg_defect_report_2026q1_0000",
    heading_path = ["# Q1 Defect Report", "## Line A"],
    content      = "Line A produced 450 units...",
    char_start   = 42, char_end = 120,
  )
```

**`embed_text()` for the chunk:**
```
"# Q1 Defect Report > ## Line A\n\nLine A produced 450 units..."
           ↑ breadcrumb prepended
```

**Qdrant upsert point:**
```json
{
  "id":     "c4a3d8f1-...",   // uuid5 of "mfg_defect_report_2026q1_0000"
  "vector": [0.023, -0.011, ...],  // BGE-M3 embedding of embed_text()
  "payload": {
    "tenant_id":          "acme-corp",
    "document_id":        "mfg_defect_report_2026q1",
    "chunk_id":           "mfg_defect_report_2026q1_0000",
    "category":           "manufacturing_data",
    "heading_path":       "# Q1 Defect Report > ## Line A",
    "char_start":         42,
    "char_end":           120,
    "permitted_purposes": "customer_support,audit",
    "content":            "Line A produced 450 units..."
  }
}
```

Collection: `manufacturing_docs`

---

### Searching for the document

```
SearchRequest:
  tenant_id = "acme-corp"
  query     = "What are the defect rates for Line A?"
  user.allowed_categories = ["manufacturing_data"]
  purpose   = "audit"
```

**Permission → collection resolution:**
```
_resolve_permissions() → ["manufacturing_data"]
_collections_for_categories(["manufacturing_data"]) → ["manufacturing_docs"]
```

**Router (MR-01):**
```
_RE_FACTUAL matches "what are" → scores[FACTUAL] += 0.5
_COLLECTION_DOMAINS["manufacturing_docs"] contains "defect", "line"
  → hits["manufacturing_docs"] = 2
  → hits["customer_docs"] = 0, hits["hr_docs"] = 0

selected = ["manufacturing_docs"]
excluded = ["customer_docs", "hr_docs"]
strategy = "vector_only"  (FACTUAL intent)
```

**`vector_search("manufacturing_docs", vec, {"tenant_id":"acme-corp"}, 50)`:**
```
Qdrant applies:
  must: [tenant_id == "acme-corp"]   ← pre-filter before ANN
  → only "acme-corp" chunks in HNSW traversal
  → cosine similarity against query vector
  → returns top 50 by score
```

**Purpose filter:**
```
purpose = "audit"
result.metadata["permitted_purposes"] = "customer_support,audit"
"audit" in "customer_support,audit".split(",") → True → allowed
```

**Final result:**
```json
{
  "document_id": "mfg_defect_report_2026q1",
  "chunk_id":    "mfg_defect_report_2026q1_0000",
  "heading_path":"# Q1 Defect Report > ## Line A",
  "score":       0.87,
  "content":     "Line A produced 450 units..."
}
```

---

## Summary: partitioning decisions at each stage

| Stage | Partitioning key | Mechanism |
|---|---|---|
| `IndexRequest.category` | data category | routes upsert to correct collection |
| `chunk.stable_uuid()` | chunk identity | deterministic Qdrant point ID (upsert idempotent) |
| `chunk.heading_path` | document structure | breadcrumb in `embed_text()` |
| `chunk.char_start / char_end` | character offset | exact source location for attribution |
| `tenant_id` filter | tenant isolation | Qdrant `must` pre-filter on every query |
| RBAC `allowed_categories` | category access | limits `permission_collections` before routing |
| Router `_select_collections()` | domain relevance | further narrows within permitted collections |
| `permitted_purposes` payload | purpose access | `_filter_by_purpose()` post-retrieval filter |

---

## Related documents

- `navigator/docs/hybrid-reranking.md` — RRF fusion and cross-encoder reranking
- `navigator/docs/CONFIGURATION.md` — full config schema with defaults
- `docs/12_module_navigator_srs_v3.md` — full Navigator SRS
- `docs/21_cross_multi_tenancy_srs.md` — cross-module multi-tenancy requirements
- `docs/31_modular_rag_requirements.md` — MR-01 routing specification
