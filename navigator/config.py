from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    rest_port: int = 8082
    grpc_port: int = 9092
    workers: int = 4


class HNSWConfig(BaseModel):
    m: int = 16
    ef_construct: int = 200
    ef_search: int = 128


class CollectionConfig(BaseModel):
    vector_size: int = 1024
    distance: str = "cosine"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)


class EmbedCacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000


class EmbedderConfig(BaseModel):
    type: str = "local"           # "local" = sentence-transformers; "bge_http" = remote
    model_name: str = "BAAI/bge-m3"
    endpoint: str = "http://localhost:8000"
    batch_size: int = 16
    max_length: int = 512
    cache: EmbedCacheConfig = Field(default_factory=EmbedCacheConfig)


class RerankerConfig(BaseModel):
    enabled: bool = True
    type: str = "local"           # "local" = cross-encoder; "bge_http" = remote
    model_name: str = "BAAI/bge-reranker-v2-m3"
    endpoint: str = "http://localhost:8001"
    batch_size: int = 16
    max_length: int = 512


class VectorDBConfig(BaseModel):
    type: str = "qdrant"
    hosts: list[str] = Field(default_factory=lambda: ["localhost:6333"])
    collections: dict[str, CollectionConfig] = Field(default_factory=dict)


class SearchDefaultsConfig(BaseModel):
    top_k: int = 10
    over_fetch_multiplier: int = 5
    use_hybrid: bool = True
    use_reranking: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    min_score: float = 0.5
    timeout_ms: int = 500


class VaultConfig(BaseModel):
    enabled: bool = False
    endpoint: str = "http://localhost:8081"
    cache_permissions: bool = True
    permission_ttl_seconds: int = 300


class CacheConfig(BaseModel):
    type: str = "redis"
    url: str = "redis://localhost:6379"
    query_cache_ttl_seconds: int = 3600
    permission_cache_ttl_seconds: int = 300


class EventsConfig(BaseModel):
    nats_url: str = ""
    enabled: bool = True


class PeerConfig(BaseModel):
    id: str
    endpoint: str
    topic_affinity: list[str] = Field(default_factory=list)
    capability: str = "search"   # "search" | "agent"


class FederationConfig(BaseModel):
    confidence_threshold: float = 0.70
    routing_threshold: float = 0.40
    max_peers_per_query: int = 3
    max_depth: int = 2
    peer_timeout_ms: int = 2000
    rrf_k: float = 60.0
    peers: list[PeerConfig] = Field(default_factory=list)


class LoopConfig(BaseModel):
    max_iterations: int = 3
    loop_timeout_ms: float = 500.0
    quality_threshold: float = 0.60
    coverage_threshold: float = 0.40
    uncertain_low: float = 0.45


class RouterConfig(BaseModel):
    routing_threshold: float = 0.25
    # FR-MR-01-003: when True, collection selection scores each collection by
    # cosine similarity between the query embedding and the collection's topic
    # centroid (maintained at index time). When False, the keyword proxy is used.
    use_embedding_affinity: bool = False
    # Per-tenant routing_threshold overrides (tenant_id → threshold).
    tenant_thresholds: dict[str, float] = Field(default_factory=dict)
    # Intent-classification regex overrides. Empty → router module defaults.
    analytical_pattern: str = ""
    procedural_pattern: str = ""
    multi_hop_pattern: str = ""
    factual_pattern: str = ""


class HyDEConfig(BaseModel):
    enabled: bool = False          # MR-02-002: opt-in
    max_words: int = 12            # only activate for queries shorter than this
    timeout_ms: float = 2000.0     # timeout for LLM-based hypothetical generation
    llm_endpoint: str = ""         # if set, call a local LLM (Ollama-compatible); else use template


class StalenessConfig(BaseModel):
    enabled: bool = True
    threshold_days: int = 7        # MR-05-004: chunks older than this are flagged


class ModularRAGConfig(BaseModel):
    enabled: bool = False          # opt-in; linear path when False
    loop: LoopConfig = Field(default_factory=LoopConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    staleness: StalenessConfig = Field(default_factory=StalenessConfig)


class LocalLLMConfig(BaseModel):
    provider: str = "ollama"             # "ollama" | "llamacpp" | "custom_http"
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2:8b"
    max_tokens: int = 2048
    timeout_seconds: int = 30


class AgentConfig(BaseModel):
    local_llm: LocalLLMConfig = Field(default_factory=LocalLLMConfig)


class ConnectorConfig(BaseModel):
    enabled: bool = False
    type: str = "jsonl"           # "jsonl" | "directory" | "rest"
    path: str = ""                # jsonl file path or directory path
    endpoint: str = ""            # REST pull endpoint base URL
    auth_header: str = ""         # Authorization header value (from Vault KMS)
    category: str = ""            # data category for indexed documents
    timeout_seconds: float = 10.0
    poll_interval_seconds: int = 300  # for REST pull; 0 = no auto-poll


class DecomposerConfig(BaseModel):
    enabled: bool = True
    max_sub_queries: int = 4
    # Sub-query split regex overrides. Empty → decomposer module defaults.
    conjunction_en_pattern: str = ""
    conjunction_kr_pattern: str = ""
    temporal_pattern: str = ""
    sentence_separator_pattern: str = ""


class TokenRewriterConfig(BaseModel):
    # Vault-token detection regex (2 groups: kind, hex suffix).
    # Empty → token_rewriter module default.
    token_pattern: str = ""


class ChunkingConfig(BaseModel):
    # Markdown structural regex overrides used by the chunker.
    # Empty → chunker module defaults.
    heading_pattern: str = ""
    table_row_pattern: str = ""
    fence_pattern: str = ""
    link_pattern: str = ""


class Config(BaseModel):
    version: str = "3.0"
    mode: str = "search"              # "search" | "federation" | "agent"
    instance_id: str = ""             # unique ID for loop-prevention; defaults to hostname
    server: ServerConfig = Field(default_factory=ServerConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    search_defaults: SearchDefaultsConfig = Field(default_factory=SearchDefaultsConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    modular_rag: ModularRAGConfig = Field(default_factory=ModularRAGConfig)
    connector: ConnectorConfig = Field(default_factory=ConnectorConfig)
    decomposer: DecomposerConfig = Field(default_factory=DecomposerConfig)
    token_rewriter: TokenRewriterConfig = Field(default_factory=TokenRewriterConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    @classmethod
    def load(cls, path: str) -> "Config":
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls.model_validate(data or {})
        except FileNotFoundError:
            return cls()
