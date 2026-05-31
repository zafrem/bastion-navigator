from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class UserContext(BaseModel):
    user_id: str = ""
    department: str = ""
    roles: list[str] = Field(default_factory=list)
    allowed_categories: list[str] = Field(default_factory=list)


class SearchOptions(BaseModel):
    top_k: int = 0
    over_fetch: int = 0
    use_reranking: bool = False
    use_hybrid: bool = False
    vector_weight: float = 0.0
    bm25_weight: float = 0.0
    filters: dict[str, str] = Field(default_factory=dict)
    min_score: float = 0.0
    timeout_ms: int = 0
    strategy: str = ""          # override router selection (MR-01)
    use_hyde: bool = False      # HyDE embedding for short factual/procedural queries (MR-02-002)


class SearchRequest(BaseModel):
    request_id: str = ""
    tenant_id: str = ""
    query: str
    user: Optional[UserContext] = None
    options: Optional[SearchOptions] = None
    purpose: str = "customer_support"   # MR-04-002: declared access purpose


class SearchResult(BaseModel):
    document_id: str = ""
    content: str = ""
    score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)
    category: str = ""
    # MR-05-002: source attribution provenance fields
    chunk_id: str = ""
    heading_path: str = ""
    char_start: int = 0
    char_end: int = 0
    last_indexed: str = ""


class SearchMetadata(BaseModel):
    total_candidates: int = 0
    filtered_out: int = 0
    final_count: int = 0
    used_cache: bool = False
    strategy: str = ""


class SearchResponse(BaseModel):
    request_id: str = ""
    results: list[SearchResult] = Field(default_factory=list)
    metadata: SearchMetadata = Field(default_factory=SearchMetadata)
    processing_time_ms: float = 0.0


class EmbedRequest(BaseModel):
    request_id: str = ""
    text: str


class EmbedResponse(BaseModel):
    request_id: str = ""
    embedding: list[float]
    dim_count: int = 0


class BatchEmbedRequest(BaseModel):
    request_id: str = ""
    texts: list[str]


class BatchEmbedResponse(BaseModel):
    request_id: str = ""
    embeddings: list[list[float]]


class RerankRequest(BaseModel):
    request_id: str = ""
    query: str
    candidates: list[SearchResult]
    top_k: int = 0


class RerankResponse(BaseModel):
    request_id: str = ""
    results: list[SearchResult]


class BatchSearchRequest(BaseModel):
    request_id: str = ""
    queries: list[SearchRequest]


class BatchSearchResponse(BaseModel):
    request_id: str = ""
    results: list[SearchResponse]


class CollectionInfo(BaseModel):
    name: str
    vector_count: int = 0
    dimensions: int = 0
    status: str = "green"


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class HealthStatus(BaseModel):
    status: str
    checks: dict[str, str]
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    error: str
    code: int
    trace_id: str = ""


PERMITTED_PURPOSES = [
    "customer_support",
    "audit",
    "hr_analytics",
    "product_development",
    "legal",
    "training_data",
]


class IndexRequest(BaseModel):
    document_id: str
    tenant_id: str = ""
    category: str = ""
    title: str = ""
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    permitted_purposes: list[str] = Field(
        default_factory=lambda: ["customer_support"]
    )  # MR-04-001: which purposes may access this document
    mime_type: str = "text/markdown"   # MR-06-004: profile selection
    source_version: str = ""           # MR-06-003: from source system if available


class IndexResponse(BaseModel):
    document_id: str
    chunk_count: int
    chunk_ids: list[str] = Field(default_factory=list)
    content_hash: str = ""             # MR-06-003: SHA-256 of indexed content
    was_updated: bool = False          # MR-06-002: True when existing chunks were replaced


# ─── Delta indexing (MR-06-002) ───────────────────────────────────────────────

class DeltaIndexRequest(BaseModel):
    """Re-index a document only when its content hash has changed."""
    document_id: str
    tenant_id: str = ""
    category: str = ""
    title: str = ""
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    permitted_purposes: list[str] = Field(default_factory=lambda: ["customer_support"])
    mime_type: str = "text/markdown"
    source_version: str = ""
    force: bool = False   # skip hash check and always re-index


class DeltaIndexResponse(BaseModel):
    document_id: str
    indexed: bool            # False = skipped (content unchanged)
    chunk_count: int = 0
    old_chunk_count: int = 0
    content_hash: str = ""


# ─── Data steward — purposes update (MR-04-004) ───────────────────────────────

class UpdatePurposesRequest(BaseModel):
    """Steward updates permitted_purposes on an already-indexed document."""
    document_id: str
    tenant_id: str = ""
    collection: str = ""   # Qdrant collection; inferred from category when empty
    permitted_purposes: list[str]
    steward_user_id: str = ""


class UpdatePurposesResponse(BaseModel):
    document_id: str
    collection: str
    chunks_updated: int
    permitted_purposes: list[str]


# ─── Federation / Agent models (doc 22) ──────────────────────────────────────

class AgentGenerateRequest(BaseModel):
    query: str
    context: list[SearchResult] = Field(default_factory=list)
    max_tokens: int = 500
    tenant_id: str = ""


class AgentGenerateResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    model: str = ""
    confidence: float = 0.0
