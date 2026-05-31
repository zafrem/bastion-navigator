# Bastion-Navigator API Reference

This document provides detailed information about the API interfaces provided by Bastion-Navigator.

## 1. REST API

The REST API is available on port `8080` by default. All requests and responses use JSON.

### 1.1 Search

#### Post `/v1/navigator/search`
Perform a standard vector or hybrid search.

**Request Body:**
```json
{
  "request_id": "uuid-string",
  "tenant_id": "tenant-01",
  "query": "What are the security policies for customer data?",
  "user": {
    "user_id": "user-123",
    "department": "IT",
    "roles": ["admin"],
    "allowed_categories": ["customer_data", "general"]
  },
  "options": {
    "top_k": 10,
    "use_reranking": true,
    "use_hybrid": true
  }
}
```

#### Post `/v1/navigator/search/hybrid`
Shortcut for performing a hybrid search (vector + BM25). Equivalent to `/search` with `use_hybrid: true`.

#### Post `/v1/navigator/search/batch`
Perform multiple searches in a single request.

---

### 1.2 Embedding

#### Post `/v1/navigator/embed`
Generate a vector embedding for a single text string.

**Request Body:**
```json
{
  "request_id": "uuid-string",
  "text": "Sample text to embed"
}
```

#### Post `/v1/navigator/embed/batch`
Generate embeddings for multiple text strings.

---

### 1.3 Reranking

#### Post `/v1/navigator/rerank`
Re-score and re-order a set of candidates based on a query.

**Request Body:**
```json
{
  "request_id": "uuid-string",
  "query": "Original search query",
  "candidates": [
    {
      "document_id": "doc-01",
      "content": "Document content...",
      "score": 0.85
    }
  ],
  "top_k": 5
}
```

---

### 1.4 Management & Health

- **GET `/v1/health`**: General health status.
- **GET `/v1/health/live`**: Liveness probe.
- **GET `/v1/health/ready`**: Readiness probe.
- **GET `/v1/metrics`**: Prometheus metrics.
- **GET `/v1/navigator/collections`**: List available vector collections.
- **GET `/v1/navigator/collections/{name}`**: Get details of a specific collection.

---

## 2. gRPC API

The gRPC API is available on port `9090` by default.

### Service: `bastion-rag.navigator.v1.NavigatorService`

#### RPC Methods

| Method | Request Type | Response Type | Description |
|---|---|---|---|
| `Search` | `SearchRequest` | `SearchResponse` | Primary search operation |
| `HybridSearch` | `HybridSearchRequest` | `SearchResponse` | Search with vector + BM25 |
| `BatchSearch` | `BatchSearchRequest` | `BatchSearchResponse` | Multiple concurrent searches |
| `Embed` | `EmbedRequest` | `EmbedResponse` | Generate single embedding |
| `BatchEmbed` | `BatchEmbedRequest` | `BatchEmbedResponse` | Generate multiple embeddings |
| `Rerank` | `RerankRequest` | `RerankResponse` | Re-score search results |
| `GetCollections`| `CollectionsRequest` | `CollectionsResponse` | List vector collections |
| `Health` | `HealthRequest` | `HealthResponse` | Check service health |

### Data Models (Protobuf)

Refer to `proto/navigator/v1/navigator.proto` for full message definitions.

---

## 3. Common Error Codes

| HTTP Code | gRPC Code | Meaning |
|---|---|---|
| 200 | OK | Request successful |
| 400 | INVALID_ARGUMENT | Malformed request or invalid parameters |
| 401 | UNAUTHENTICATED | Missing or invalid credentials |
| 403 | PERMISSION_DENIED | User does not have access to the resource |
| 404 | NOT_FOUND | Collection or resource not found |
| 500 | INTERNAL | Server error |
| 503 | UNAVAILABLE | Dependent service (Qdrant, Vault) is down |
