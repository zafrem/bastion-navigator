# Bastion-Navigator (Module C)

## Overview
**Bastion-Navigator** is the search and ranking layer of the Bastion-RAG Security Governance Framework. It provides high-performance vector search, hybrid retrieval, and re-ranking capabilities while ensuring tenant-level isolation and access-aware filtering.

Navigator integrates with **Qdrant** for vector storage and uses **BGE-M3** embeddings for multilingual support (Korean & English).

## Key Features
- **Vector Search:** High-speed ANN search using HNSW indexing.
- **Hybrid Search:** Combines vector similarity with BM25 lexical search using Reciprocal Rank Fusion (RRF).
- **Reranking:** Refines search results using Cross-encoder models (BGE-reranker).
- **Tenant Partitioning:** Ensures logical isolation of data through metadata filtering.
- **Access-Aware Filtering:** Integrates with **Bastion-Vault** to filter search results based on user permissions and data categories.
- **Multilingual Support:** Optimized for 50/50 Korean and English document sets.

## Architecture
Navigator is composed of several internal components:
- **Orchestrator:** Manages the search pipeline (Embed -> Search -> Rerank -> Filter).
- **Embedder:** Interface for generating vector representations of queries.
- **Searcher:** Integration with Qdrant vector database.
- **Reranker:** Interface for scoring and re-ordering results.
- **Vault Client:** Integration with Module B for access control and anonymization.

## Getting Started

### Prerequisites
- Go 1.21+
- Qdrant 1.10+
- Embedding & Reranker services (BGE models)
- Redis (Optional, for caching)

### Build
To build the `navigator-cli` binary:
```bash
make build
```

### Installation & Execution
You can run Navigator in standalone mode for testing or with a configuration file for production.

**Standalone Mode (Testing):**
```bash
make run-standalone
```

**Production Mode:**
```bash
make run-server
```

**Interactive Search (CLI):**
```bash
make interactive
```

**Docker:**
```bash
make docker-up
```

### Development
- `make test`: Run all tests.
- `make lint`: Run golangci-lint.
- `make generate`: Generate gRPC code from proto files.

## API Interfaces
Navigator provides multiple interfaces:
- **gRPC:** Port 9090 (Default)
- **REST API:** Port 8080 (Default)
- **Metrics:** Port 9091 (Prometheus)

## Documentation
- [Design Document](DESIGN.md)
- [API Reference](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Integration & Connection Guide](docs/INTEGRATION.md)
- [SRS Document](docs/bastion_navigator_srs_v1.0_en.md)

**Technical deep-dives (code-based):**
- [Hybrid Search & Reranking](docs/hybrid-reranking.md)
- [Logical Partitioning](docs/logical-partitioning.md)
- [Source Connectors & Delta Indexing](docs/source-connectors.md)
- [Sub-Query Decomposition](docs/sub-query-decomposition.md)

## License
Apache License 2.0
