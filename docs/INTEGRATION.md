# Integration & Connection Guide

This document describes how Bastion-Navigator can be integrated into the broader Bastion-RAG pipeline and the different connection modes supported.

## 1. Connection Modes

Bastion-Navigator supports three primary integration patterns:

### 1.1 Serial Pipeline (Standard)
In the standard Bastion-RAG pipeline, modules are called sequentially:
`Sentinel (Module A) -> Vault (Module B) -> Navigator (Module C) -> Anchor (Module E) -> LLM`

In this mode, `Vault` resolves user permissions and PII anonymization before passing the request to `Navigator`.

### 1.2 Inline / Middleware Connection
Navigator can operate in an **inline** fashion where it receives "pre-resolved" context from upstream modules. This is supported through the `UserContext` field in the search request.

If the `allowed_categories` list is populated in the request, Navigator will **skip** the direct call to Vault and use the provided categories for filtering. This reduces latency and allows Navigator to act as a specialized retrieval middleware within a pre-authorized flow.

### 1.3 Standalone / Replacement Mode
Navigator can run as a standalone service, independent of the rest of the Bastion-RAG framework.
- **Standalone:** By setting `vault.enabled: false` in the configuration, Navigator will mock permission checks (allowing all access) and skip external dependencies.
- **Replacement:** Navigator's gRPC and REST interfaces are designed to be compatible with standard RAG retrieval patterns, allowing it to replace existing search layers without significant code changes.

---

## 2. Protocol Verification

Navigator maintains strict protocol consistency between its two primary interfaces:

### 2.1 gRPC (System-to-System)
- **Port:** 9090
- **Protocol:** Protobuf / gRPC
- **Usage:** Recommended for internal Bastion-RAG module communication to minimize serialization overhead.

### 2.2 REST API (External/App)
- **Port:** 8080
- **Protocol:** JSON / HTTP
- **Usage:** Recommended for external applications, frontend integrations, and languages without mature gRPC support.

### 2.3 Shared Data Models
Both interfaces use the same underlying data models defined in the `internal/models` package. This ensures that:
- Search parameters (`top_k`, `use_hybrid`, etc.) behave identically across protocols.
- Scoring and metadata results are consistent.
- Validation logic is applied uniformly.

---

## 3. Configuration Scenarios

### 3.1 High-Security (Full Pipeline)
```yaml
vault:
  enabled: true
  endpoint: "http://vault:8080"
search_defaults:
  over_fetch_multiplier: 5 # More candidates for strict filtering
```

### 3.2 High-Performance (Pre-resolved)
Upstream module (e.g., a custom orchestrator) resolves permissions once and passes them to Navigator.
```json
{
  "query": "...",
  "user": {
    "allowed_categories": ["customer_data"]
  }
}
```

### 3.3 Dev/Test (Standalone)
```yaml
vault:
  enabled: false
vector_db:
  hosts: ["localhost:6333"]
```
