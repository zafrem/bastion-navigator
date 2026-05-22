# Configuration Guide

Bastion-Navigator uses a YAML configuration file to define its behavior, server ports, and integrations with external services.

## File Structure

The default configuration file is located at `config/config.yaml`.

### 1. Server Configuration
```yaml
server:
  rest_port: 8080   # Port for REST API
  grpc_port: 9090   # Port for gRPC API
  workers: 4        # Number of worker goroutines for search processing
```

### 2. Vector Database (Qdrant)
```yaml
vector_db:
  type: qdrant
  hosts:
    - qdrant:6333
  collections:
    customer_docs:
      vector_size: 1024
      distance: cosine
      hnsw:
        m: 16
        ef_construct: 200
        ef_search: 128
```

### 3. Embedder & Reranker
Navigator relies on external BGE model services.
```yaml
embedder:
  type: bge_m3
  endpoint: http://embedder:8000
  batch_size: 16
  cache:
    enabled: true
    ttl: 1h

reranker:
  enabled: true
  type: bge_reranker
  endpoint: http://reranker:8001
```

### 4. Search Defaults
These values are used when search options are not specified in the request.
```yaml
search_defaults:
  top_k: 10
  over_fetch_multiplier: 5
  use_hybrid: true
  use_reranking: true
  vector_weight: 0.7
  bm25_weight: 0.3
  min_score: 0.5
  timeout_ms: 500
```

### 5. External Integrations
- **Vault:** Integration with Module B for access control.
- **Tracker:** Integration with Module D for audit logging.
- **Cache:** Redis-based caching for search results and permissions.

## Environment Variables
(Planned feature) Support for overriding YAML values with environment variables (e.g., `NAVIGATOR_SERVER_PORT`).
