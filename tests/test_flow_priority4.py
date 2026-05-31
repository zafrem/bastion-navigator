"""End-to-end flow test for Priority 4 features through the real Orchestrator.

Uses a functional in-memory searcher (not bare mocks) so that index → search →
delta re-index → steward purpose update genuinely compose: documents written by
one step are visible to the next, deletes really remove chunks, and payload
patches really mutate stored state.

This is the "overall flow" test for the Navigator enterprise-integration layer.
"""
from __future__ import annotations

from navigator.config import Config
from navigator.connector import JsonlConnector, Document
from navigator.models import (
    IndexRequest, DeltaIndexRequest, SearchRequest, SearchOptions,
    UpdatePurposesRequest, UserContext,
)
from navigator.orchestrator import Orchestrator


# ─── Functional in-memory searcher ────────────────────────────────────────────

class InMemorySearcher:
    """A real (if simple) searcher: stores points, filters by payload, supports
    the full delta-index surface (delete/count/set_payload)."""

    def __init__(self):
        # collection -> list[point dict]
        self.collections_data: dict[str, list[dict]] = {}

    def ensure_collection(self, name: str, vector_size: int = 768) -> None:
        self.collections_data.setdefault(name, [])

    def upsert(self, collection: str, points: list[dict]) -> None:
        self.collections_data.setdefault(collection, [])
        # Replace by id if present (idempotent upsert).
        existing = {p["id"]: i for i, p in enumerate(self.collections_data[collection])}
        for p in points:
            if p["id"] in existing:
                self.collections_data[collection][existing[p["id"]]] = p
            else:
                self.collections_data[collection].append(p)

    def _matches(self, payload: dict, filters: dict) -> bool:
        return all(str(payload.get(k, "")) == str(v) for k, v in filters.items())

    def vector_search(self, collection, vector, filters, top_k, min_score=0.0):
        from navigator.models import SearchResult
        pts = self.collections_data.get(collection, [])
        out = []
        for p in pts:
            pl = p["payload"]
            if filters and not self._matches(pl, filters):
                continue
            prov = {"content", "document_id", "chunk_id", "heading_path",
                    "char_start", "char_end", "last_indexed"}
            out.append(SearchResult(
                document_id=pl.get("document_id", ""),
                content=pl.get("content", ""),
                score=0.9,
                chunk_id=pl.get("chunk_id", ""),
                last_indexed=pl.get("last_indexed", ""),
                # SearchResult.metadata is dict[str, str]; coerce like a real
                # provenance extraction would (Qdrant payloads are stringified).
                metadata={k: str(v) for k, v in pl.items() if k not in prov},
            ))
        return out[:top_k]

    def sparse_search(self, collection, query, filters, top_k):
        return self.vector_search(collection, [], filters, top_k)

    def collections(self):
        from navigator.models import CollectionInfo
        return [CollectionInfo(name=n) for n in self.collections_data]

    def delete_by_document(self, collection: str, document_id: str) -> int:
        pts = self.collections_data.get(collection, [])
        before = len(pts)
        self.collections_data[collection] = [
            p for p in pts if p["payload"].get("document_id") != document_id
        ]
        return before - len(self.collections_data[collection])

    def count_by_document(self, collection: str, document_id: str) -> int:
        return sum(
            1 for p in self.collections_data.get(collection, [])
            if p["payload"].get("document_id") == document_id
        )

    def set_payload(self, collection: str, document_id: str, patch: dict) -> int:
        n = 0
        for p in self.collections_data.get(collection, []):
            if p["payload"].get("document_id") == document_id:
                p["payload"].update(patch)
                n += 1
        return n


class IdentityEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class PassthroughReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class AllowAllVault:
    def allowed_categories(self, user_id):
        return None


def _orch() -> tuple[Orchestrator, InMemorySearcher]:
    searcher = InMemorySearcher()
    orch = Orchestrator(
        cfg=Config(),
        embedder=IdentityEmbedder(),
        searcher=searcher,
        reranker=PassthroughReranker(),
        vault=AllowAllVault(),
    )
    return orch, searcher


# ─── Flow 1: connector → index → search ───────────────────────────────────────

def test_flow_connector_index_search(tmp_path):
    # A connector yields a document; we index it; then it is searchable.
    f = tmp_path / "docs.jsonl"
    f.write_text(
        '{"id":"leave-policy","content":"# Leave Policy\\n\\nEmployees get 15 days annual leave.",'
        '"category":"hr_data","title":"Leave"}\n'
    )
    docs = list(JsonlConnector(str(f)).list_documents())
    assert len(docs) == 1

    orch, searcher = _orch()
    doc = docs[0]
    # Index into the COLLECTION name ("hr_docs"); search resolves the
    # data-category label "hr_data" → collection "hr_docs" (see
    # orchestrator._CATEGORY_TO_COLLECTION).
    resp = orch.index_document(IndexRequest(
        document_id=doc.id, tenant_id="acme", category="hr_docs",
        title=doc.title, content=doc.content, mime_type=doc.mime_type,
    ))
    assert resp.chunk_count >= 1
    assert resp.content_hash  # MR-06-003

    # Search returns the indexed chunk.
    sresp = orch.search(SearchRequest(
        request_id="s1", tenant_id="acme", query="annual leave",
        user=UserContext(user_id="alice", allowed_categories=["hr_data"]),
        options=SearchOptions(top_k=5),
    ))
    assert len(sresp.results) >= 1
    assert "leave" in sresp.results[0].content.lower()


# ─── Flow 2: delta re-index — unchanged skips, changed replaces ───────────────

def test_flow_delta_reindex_lifecycle():
    orch, searcher = _orch()

    base = dict(document_id="d1", tenant_id="acme", category="hr_data")

    # Initial index.
    orch.index_document(IndexRequest(**base, content="Version one of the document."))
    assert searcher.count_by_document("hr_data", "d1") >= 1

    # Delta with identical content → skipped (indexed=False).
    same = orch.delta_index_document(DeltaIndexRequest(**base, content="Version one of the document."))
    assert same.indexed is False

    # Delta with changed content → re-index (old chunks deleted, new inserted).
    changed = orch.delta_index_document(DeltaIndexRequest(
        **base, content="Version two has completely different and longer content here."))
    assert changed.indexed is True
    assert changed.old_chunk_count >= 1
    assert changed.content_hash != same.content_hash

    # The store reflects ONLY the new version (no mixed-version chunks).
    pts = searcher.collections_data["hr_data"]
    contents = " ".join(p["payload"]["content"] for p in pts)
    assert "Version two" in contents
    assert "Version one" not in contents


# ─── Flow 3: steward updates permitted_purposes (no re-embed) ─────────────────

def test_flow_steward_purpose_update():
    orch, searcher = _orch()

    orch.index_document(IndexRequest(
        document_id="d2", tenant_id="acme", category="customer_data",
        content="Customer record content.",
        permitted_purposes=["customer_support"],
    ))

    # Steward broadens purposes.
    resp = orch.update_document_purposes(UpdatePurposesRequest(
        document_id="d2", tenant_id="acme", collection="customer_data",
        permitted_purposes=["customer_support", "audit"],
        steward_user_id="steward.kim",
    ))
    assert resp.chunks_updated >= 1
    assert resp.permitted_purposes == ["customer_support", "audit"]

    # The stored payload now carries the new purposes (mutation, not re-embed).
    pts = searcher.collections_data["customer_data"]
    for p in pts:
        if p["payload"]["document_id"] == "d2":
            assert p["payload"]["permitted_purposes"] == "customer_support,audit"


# ─── Flow 4: purpose filter excludes disallowed documents at search time ──────

def test_flow_purpose_filter_end_to_end():
    orch, searcher = _orch()

    # Two docs in the customer_docs collection: one allows "audit", one not.
    # (search resolves allowed_categories=["customer_data"] → "customer_docs")
    orch.index_document(IndexRequest(
        document_id="audit-doc", tenant_id="acme", category="customer_docs",
        content="Audit-permitted document about balances.",
        permitted_purposes=["customer_support", "audit"],
    ))
    orch.index_document(IndexRequest(
        document_id="cs-doc", tenant_id="acme", category="customer_docs",
        content="Support-only document about balances.",
        permitted_purposes=["customer_support"],
    ))

    # Search with purpose="audit" should exclude the cs-only doc.
    resp = orch.search(SearchRequest(
        request_id="s2", tenant_id="acme", query="balances",
        purpose="audit",
        user=UserContext(user_id="auditor", allowed_categories=["customer_data"]),
        options=SearchOptions(top_k=10),
    ))
    ids = {r.document_id for r in resp.results}
    assert "audit-doc" in ids
    assert "cs-doc" not in ids


# ─── Flow 5: CSV profile chunking through index path ──────────────────────────

def test_flow_csv_profile_indexing():
    orch, searcher = _orch()
    csv_content = "name,dept,role\nAlice,HR,manager\nBob,Eng,staff\nCarol,Eng,lead"
    resp = orch.index_document(IndexRequest(
        document_id="employees", tenant_id="acme", category="hr_data",
        content=csv_content, mime_type="text/csv",
    ))
    # One chunk per data row (3 rows).
    assert resp.chunk_count == 3
    pts = searcher.collections_data["hr_data"]
    # Header is present on each row chunk.
    assert all("name,dept,role" in p["payload"]["content"] for p in pts)
