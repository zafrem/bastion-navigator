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


class SearchRequest(BaseModel):
    request_id: str = ""
    tenant_id: str = ""
    query: str
    user: Optional[UserContext] = None
    options: Optional[SearchOptions] = None


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


class IndexRequest(BaseModel):
    document_id: str
    tenant_id: str = ""
    category: str = ""
    title: str = ""
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    document_id: str
    chunk_count: int
    chunk_ids: list[str] = Field(default_factory=list)


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
