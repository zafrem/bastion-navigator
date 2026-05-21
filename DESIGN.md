# Bastion-Navigator Design Document

## 1. Introduction
Navigator is Module C of the Bastion framework, responsible for retrieval operations in the RAG pipeline. It ensures that only relevant and authorized documents are retrieved for a given query.

## 2. System Architecture

### 2.1 High-Level Component Diagram
```
  [API Layer (gRPC/REST/CLI)]
             |
             v
      [Orchestrator]
             |
      +------+------+------+
      |      |      |      |
 [Embedder] [Searcher] [Reranker] [Vault Client]
             |             |
         [Qdrant]    [BGE-Reranker]
```

### 2.2 Component Responsibilities
- **Orchestrator:** The core logic that coordinates retrieval. It converts queries to embeddings, performs the search, reranks results, and applies security filters.
- **Embedder:** Wraps the embedding model service (BGE-M3). Supports batch embedding and query embedding.
- **Searcher:** Interfaces with Qdrant. Handles collection management, payload filtering, and hybrid search queries.
- **Reranker:** Uses a Cross-encoder to re-score the top-K results from the initial search to improve precision.
- **Vault Client:** Queries Module B (Vault) to retrieve user permissions and ensure that retrieved documents match the user's access level.

## 3. Data Flow

### 3.1 Search Flow
1. **Request:** Receive search request with query and user context.
2. **Embedding:** Convert query to a 1024-dimensional vector using BGE-M3.
3. **Primary Search:**
   - Vector search in Qdrant with HNSW.
   - Keyword search (BM25) if hybrid mode is enabled.
   - Apply pre-filters (tenant_id, allowed_categories).
4. **Fusion:** Combine results using Reciprocal Rank Fusion (RRF).
5. **Reranking:** Re-score the top results (e.g., top 50) using BGE-reranker.
6. **Filtering:** Final check against Vault permissions.
7. **Response:** Return top-K relevant, authorized documents.

## 4. Technical Stack
- **Language:** Go 1.21
- **Vector DB:** Qdrant
- **Models:** BGE-M3 (Multilingual), BGE-reranker-v2-m3
- **Communication:** gRPC (Protobuf), REST (JSON)
- **Cache:** Redis (Optional)

## 5. Security Design
- **Logical Isolation:** Every search is scoped by `tenant_id` in the Qdrant payload.
- **Category Filtering:** Documents are tagged with security categories (e.g., `customer_data`, `hr_data`). Navigator filters these based on the `allowed_categories` list provided in the request context.
- **Integration with Module B:** Navigator relies on Vault for the authoritative source of permissions.

## 6. Performance Optimization
- **Concurrent Execution:** Embedding and Search parameters can be tuned for throughput.
- **Caching:** Search results and embeddings can be cached in Redis to reduce latency for frequent queries.
- **Efficient Filtering:** Uses Qdrant's payload indexing for fast pre-filtering.
