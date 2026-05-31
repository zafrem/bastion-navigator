# Navigator — Source Connectors & Delta Indexing

**Module:** Navigator (`navigator/connector.py`, `navigator/chunker.py`, `navigator/orchestrator.py`, `navigator/searcher.py`)
**Version:** 3.1
**Last updated:** 2026-05-31

---

## Overview

Enterprise data integration (MR-06) keeps the vector index current as source documents change. Four capabilities:

| FR | Capability | Where |
|---|---|---|
| FR-MR-06-001 | Source connector interface | `connector.py` · `SourceConnector` ABC |
| FR-MR-06-002 | Delta indexing / CDC | `orchestrator.delta_index_document()` |
| FR-MR-06-003 | Content hash storage | `index_document()` Qdrant payload |
| FR-MR-06-004 | Schema-aware chunking profiles | `chunker.py` · `chunk_by_profile()` |

All connectors feed the same pipeline: `Document → Chunker → Embedder → Qdrant upsert`.

---

## 1. The `SourceConnector` interface (FR-MR-06-001)

```python
# navigator/connector.py

class SourceConnector(ABC):
    """Abstract base for all source connectors (FR-MR-06-001)."""

    @abstractmethod
    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        """Yield documents. When since is set, yield only docs updated after that time."""

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Fetch a single document by ID, or None if not found."""

    @abstractmethod
    def document_updated_at(self, doc_id: str) -> Optional[datetime]:
        """Return the last-modified timestamp for doc_id, or None if unknown."""
```

### The `Document` domain model

```python
# navigator/connector.py

@dataclass
class Document:
    id: str
    content: str
    title: str = ""
    category: str = ""
    metadata: dict = field(default_factory=dict)
    updated_at: Optional[datetime] = None
    source_version: str = ""
    mime_type: str = "text/markdown"

    def content_hash(self) -> str:
        """SHA-256 of the document content (hex, 64 chars)."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
```

`content_hash()` is the basis for change detection (§2); `mime_type` drives profile selection (§4).

### Built-in connectors

| Connector | Source | Document ID | Change signal |
|---|---|---|---|
| `JsonlConnector` | JSONL file | `id` field per line | `updated_at` field |
| `DirectoryConnector` | filesystem directory | path relative to root | file mtime |
| `RestPullConnector` | HTTP endpoint | `id` from response | `updated_at` field |

#### JsonlConnector — wraps the existing ingestion path

```python
# navigator/connector.py

class JsonlConnector(SourceConnector):
    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        try:
            with open(self._path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("[jsonl] line %d: invalid JSON, skipped", lineno)
                        continue  # one bad line never aborts the stream
                    doc = _doc_from_dict(obj)
                    if since and doc.updated_at and doc.updated_at <= since:
                        continue  # honor the since filter
                    yield doc
        except FileNotFoundError:
            log.error("[jsonl] file not found: %s", self._path)
```

#### DirectoryConnector — filesystem watcher source

```python
# navigator/connector.py

class DirectoryConnector(SourceConnector):
    def list_documents(self, since: Optional[datetime] = None) -> Iterator[Document]:
        for root, _, files in os.walk(self._dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self._exts:   # .md/.txt/.csv/.json/.html by default
                    continue
                fpath = os.path.join(root, fname)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=timezone.utc)
                if since and mtime <= since:
                    continue  # only re-yield files modified after `since`
                rel = os.path.relpath(fpath, self._dir)
                # ... read content, infer mime_type from extension ...
                yield Document(id=rel, content=content, updated_at=mtime,
                               mime_type=_EXT_MIME.get(ext, "text/plain"))
```

#### RestPullConnector — credentials from Vault KMS (SC-09)

```python
# navigator/connector.py

class RestPullConnector(SourceConnector):
    def __init__(self, endpoint, auth_header="", timeout=10.0, page_size=100):
        self._endpoint = endpoint.rstrip("/")
        # auth_header is supplied by the caller AFTER fetching from Vault KMS.
        # Credentials are NEVER read from a config file (SC-09).
        self._headers = {"Authorization": auth_header} if auth_header else {}
        # ...

    def list_documents(self, since=None) -> Iterator[Document]:
        params = {"limit": self._page_size}
        if since:
            params["since"] = since.isoformat()
        resp = httpx.get(f"{self._endpoint}/documents", params=params,
                         headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        for obj in resp.json().get("documents", []):
            yield _doc_from_dict(obj)
```

### Factory

```python
# navigator/connector.py

def build_connector(cfg) -> Optional[SourceConnector]:
    """Build a connector from a ConnectorConfig. Returns None when disabled."""
    if not cfg or not cfg.enabled:
        return None
    t = cfg.type
    if t == "jsonl":
        return JsonlConnector(cfg.path)
    if t == "directory":
        return DirectoryConnector(cfg.path, category=cfg.category)
    if t == "rest":
        return RestPullConnector(cfg.endpoint, auth_header=cfg.auth_header,
                                 timeout=cfg.timeout_seconds)
    log.warning("[connector] unknown type %r", t)
    return None
```

Config (`config.yaml`):
```yaml
connector:
  enabled: false
  type: jsonl              # "jsonl" | "directory" | "rest"
  path: ""
  endpoint: ""
  auth_header: ""          # injected from Vault KMS at runtime (SC-09)
  category: ""
  poll_interval_seconds: 300
```

---

## 2. Delta indexing (FR-MR-06-002)

`delta_index_document` re-indexes a document only when its content has actually changed, and guarantees no mixed-version window (SC-10):

```python
# navigator/orchestrator.py

def delta_index_document(self, req: DeltaIndexRequest) -> DeltaIndexResponse:
    new_hash = hashlib.sha256(req.content.encode("utf-8")).hexdigest()
    collection = req.category or "default"

    # Retrieve the hash stored on an existing chunk's payload.
    stored_hash = self._get_stored_hash(collection, req.document_id)

    if not req.force and stored_hash and stored_hash == new_hash:
        return DeltaIndexResponse(
            document_id=req.document_id, indexed=False, content_hash=new_hash,
        )  # no change → no-op (the fast path for CDC polling)

    old_count = self._searcher.count_by_document(collection, req.document_id)

    # SC-10: delete ALL old chunks BEFORE inserting new ones.
    # No moment exists where both old and new versions are retrievable.
    self._searcher.delete_by_document(collection, req.document_id)

    index_req = IndexRequest(
        document_id=req.document_id, tenant_id=req.tenant_id, category=req.category,
        title=req.title, content=req.content, metadata=req.metadata,
        permitted_purposes=req.permitted_purposes,
        mime_type=req.mime_type, source_version=req.source_version,
    )
    index_resp = self.index_document(index_req)

    if self._publisher:
        tc = TraceContext(tenant_id=req.tenant_id)
        self._publisher.publish(event_document_reindexed(
            tc, document_id=req.document_id, collection=collection,
            old_chunk_count=old_count, new_chunk_count=index_resp.chunk_count,
            changed_sections=[], reindex_ms=0.0,
        ))

    return DeltaIndexResponse(
        document_id=req.document_id, indexed=True,
        chunk_count=index_resp.chunk_count, old_chunk_count=old_count,
        content_hash=new_hash,
    )
```

### Server-side delete by filter

`delete_by_document` uses Qdrant's `FilterSelector` so the delete happens server-side in one round-trip — there is no client-side scan-then-delete:

```python
# navigator/searcher.py

def delete_by_document(self, collection: str, document_id: str) -> int:
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, FilterSelector,
    )
    result = self._client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        ),
    )
    return getattr(result, "deleted", 0) or 0
```

---

## 3. Content hash storage (FR-MR-06-003)

Every chunk payload carries the document-level hash, the source version, and the MIME type. The hash is what `_get_stored_hash` reads back during delta checks:

```python
# navigator/orchestrator.py — inside index_document()

content_hash = _hashlib.sha256(req.content.encode("utf-8")).hexdigest()
# ...
"payload": {
    "document_id": req.document_id,
    "chunk_id": chunk.chunk_id,
    # ...
    "permitted_purposes": ",".join(req.permitted_purposes),
    "content_hash": content_hash,          # MR-06-003
    "source_version": req.source_version,  # MR-06-003
    "mime_type": req.mime_type,            # MR-06-004
    "content": chunk.content,
}
```

`IndexResponse.content_hash` and `DeltaIndexResponse.content_hash` surface it to callers so an external CDC driver can cache it and avoid sending unchanged documents at all.

---

## 4. Schema-aware chunking profiles (FR-MR-06-004)

A document's MIME type selects a chunking strategy. `profile_for_mime` maps MIME (falling back to file extension) to a profile name:

```python
# navigator/chunker.py

_MIME_PROFILE: dict[str, str] = {
    "text/markdown":    PROFILE_MARKDOWN,
    "text/plain":       PROFILE_PLAIN_TEXT,
    "text/csv":         PROFILE_STRUCTURED_CSV,
    "application/json": PROFILE_JSON_RECORD,
    "text/html":        PROFILE_HTML,
}

def profile_for_mime(mime_type: str, filename: str = "") -> str:
    if mime_type and mime_type in _MIME_PROFILE:
        return _MIME_PROFILE[mime_type]
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in _EXT_PROFILE:
            return _EXT_PROFILE[ext]
    return PROFILE_MARKDOWN  # safe default
```

| Profile | max_chars | Strategy |
|---|---|---|
| `markdown` | 1200 | Heading-boundary, table-atomic, PII split guard |
| `plain_text` | 800 | Sentence-boundary (NLTK Punkt) |
| `structured_csv` | — | One chunk per data row; header prepended |
| `json_record` | — | One chunk per top-level array element |
| `html` | 1200 | Strip tags, then markdown strategy |

### Dispatch

```python
# navigator/chunker.py

def chunk_by_profile(document_id, content, profile, metadata=None) -> list[Chunk]:
    metadata = metadata or {}
    if profile == PROFILE_STRUCTURED_CSV:
        return chunk_csv(document_id, content, metadata)
    if profile == PROFILE_JSON_RECORD:
        return chunk_json(document_id, content, metadata)
    if profile == PROFILE_HTML:
        return chunk_html(document_id, content, metadata)
    cfg = config_for_profile(profile)
    return chunk_document(document_id, content, cfg, metadata)
```

### CSV — one row per chunk, header preserved

A row in isolation is meaningless without its column names, so the header is prepended to every chunk:

```python
# navigator/chunker.py

def chunk_csv(document_id, content, metadata=None) -> list[Chunk]:
    import csv, io
    rows = list(csv.reader(io.StringIO(content)))
    if not rows:
        return chunks
    header = rows[0]
    header_text = ",".join(header)
    for idx, row in enumerate(rows[1:]):
        row_text = ",".join(row)
        combined = f"{header_text}\n{row_text}"  # header context on every chunk
        chunks.append(Chunk(chunk_id=f"{document_id}_{idx:04d}", content=combined, ...))
    return chunks
```

### JSON — one record per chunk

```python
# navigator/chunker.py

def chunk_json(document_id, content, metadata=None) -> list[Chunk]:
    import json as _json
    try:
        data = _json.loads(content)
    except _json.JSONDecodeError:
        return chunk_document(document_id, content, ChunkerConfig(max_chars=800), metadata)
    records = data if isinstance(data, list) else [data]  # array → N chunks; object → 1
    for idx, record in enumerate(records):
        text = _json.dumps(record, ensure_ascii=False)
        chunks.append(Chunk(chunk_id=f"{document_id}_{idx:04d}", content=text, ...))
    return chunks
```

### HTML — stdlib tag stripping

```python
# navigator/chunker.py

def chunk_html(document_id, content, metadata=None, cfg=None) -> list[Chunk]:
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts = []
        def handle_data(self, data):
            self._parts.append(data)
        def get_text(self):
            return " ".join(self._parts)

    stripper = _Stripper()
    stripper.feed(content)
    plain = stripper.get_text().strip()
    cfg = cfg or ChunkerConfig(max_chars=1200, overlap_chars=120)
    return chunk_document(document_id, plain, cfg, metadata or {})
```

`index_document` picks the profile automatically:

```python
# navigator/orchestrator.py — inside index_document()

profile = profile_for_mime(req.mime_type)
if profile in ("structured_csv", "json_record", "html"):
    chunks = chunk_by_profile(req.document_id, req.content, profile, dict(req.metadata))
else:
    chunks = chunk_document(req.document_id, req.content,
                            ChunkerConfig(pii_patterns=pii_patterns), dict(req.metadata))
```

---

## REST endpoints

```
POST  /v1/navigator/index          — standard index (auto-profile by mime_type)
POST  /v1/navigator/index/delta    — delta index (hash comparison, no-op if unchanged)
PATCH /v1/navigator/documents/{id}/purposes  — steward purpose update (MR-04-004)
```

---

## End-to-end trace — delta re-index of a changed Markdown file

```
1. DirectoryConnector.list_documents(since=last_run) yields:
     Document(id="policies/leave.md", content="...updated...",
              mime_type="text/markdown", updated_at=2026-05-31T09:00Z)

2. Driver calls POST /v1/navigator/index/delta with the document.

3. delta_index_document:
     new_hash      = sha256("...updated...") = "a1b2c3..."
     stored_hash   = _get_stored_hash("default", "policies/leave.md") = "f9e8d7..."
     new != stored → proceed

     old_count = count_by_document(...) = 4
     delete_by_document(...) → 4 old chunks removed (server-side filter)
     index_document(...) → profile=markdown → 5 new chunks upserted
       each chunk payload carries content_hash="a1b2c3..."

     emit event_document_reindexed(old=4, new=5)

4. Response:
     { "document_id": "policies/leave.md", "indexed": true,
       "chunk_count": 5, "old_chunk_count": 4, "content_hash": "a1b2c3..." }

5. Next poll with unchanged content:
     new_hash == stored_hash → { "indexed": false }  (no delete, no embed)
```

---

## Related documents

- `navigator/docs/logical-partitioning.md` — chunker internals, tenant/category partitioning
- `navigator/docs/sub-query-decomposition.md` — multi-hop query decomposition (MR-02-003)
- `docs/12_module_navigator_srs_v3.md` — full Navigator SRS (§11b Enterprise Data Integration)
- `docs/31_modular_rag_requirements.md` — MR-06 specification, SC-09 / SC-10 constraints
