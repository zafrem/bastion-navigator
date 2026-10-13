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

> Navigator ships as a **Python / FastAPI** service — the canonical implementation driven by the `Makefile` and `config/config.yaml`. A secondary Go implementation also lives under `cmd/navigator-cli`.

### Prerequisites
- Python 3.11+
- (Optional) Qdrant 1.10+ for vector storage — without it Navigator falls back to an in-memory mock searcher
- (Optional) Redis for the embedding cache; (Optional) NATS for event streaming
- Embedding/reranker models `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` are downloaded on first run when `embedder.type: local` — set `embedder.type: mock` in the config for a lightweight, dependency-free run

### Install
```bash
make install          # = pip install -e .
# dev extras: make install-dev   (= pip install -e ".[dev]")
```
Using a virtual environment is recommended (`python -m venv .venv && . .venv/bin/activate`).

### Run
```bash
make run              # = python -m navigator.main --config config/config.yaml
# or: make dev
```
Navigator starts the REST API on **:8082** and gRPC on **:9092**.

### Development
- `make test`   — pytest suite
- `make lint`   — ruff check
- `make format` — ruff format
- `make docker-up` / `make docker-down` — run the stack with Docker Compose

## Vault integration

When `vault.enabled` is true, Navigator asks Vault which data categories a user may read and searches only the matching collections. Vault's REST API requires a Bearer JWT, so Navigator authenticates with a service token:

```yaml
vault:
  enabled: true
  endpoint: http://vault:8081
  auth_token_file: /run/secrets/vault-token   # re-read on every call, so rotation needs no restart
  # auth_token: ""                            # or inline (avoid committing it)
```

The `NAVIGATOR_VAULT_AUTH_TOKEN` environment variable is used when neither is set. The token is an HS256 JWT signed with Vault's `auth.jwt_secret`, with Vault's issuer (default `bastion`) and an `exp` claim; a missing, forged or expired token is rejected with `401`.

- **Fails closed.** If Vault cannot be reached, rejects the token, or returns something unexpected, Navigator searches no collections instead of all of them. An empty permission list from Vault likewise means no access.
- **Per-user decisions.** Navigator sends the request's `department` and `roles` (and `tenant_id`) so Vault decides for the user, not for the service identity. A request with a `user_id` but no department or role is denied, and Vault is not asked. Several roles are unioned.
- **Categories.** Vault's `DC-01`, `DC-02` and `DC-03` map to `customer_data`, `manufacturing_data` and `hr_data`.
- `user.allowed_categories` supplied in the request still bypasses the Vault lookup (the caller is trusted to have resolved it).

## API Interfaces
- **REST API:** `:8082` — `/v1/navigator/search`, `/v1/navigator/search/hybrid`, `/v1/navigator/index`, `/v1/navigator/embed`, `/v1/navigator/rerank`, …
- **gRPC:** `:9092`
- **Health:** `GET /v1/health`
- **Metrics:** Prometheus at `GET /v1/metrics`

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
