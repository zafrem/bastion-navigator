package models

// SearchOptions controls search behaviour per-request.
type SearchOptions struct {
	TopK         int               `json:"top_k"`
	UseHybrid    bool              `json:"use_hybrid"`
	UseReranking bool              `json:"use_reranking"`
	MinScore     float64           `json:"min_score"`
	Filters      map[string]string `json:"filters,omitempty"`
}

// SearchRequest is the payload for a single vector/hybrid search.
type SearchRequest struct {
	RequestID string         `json:"request_id"`
	TenantID  string         `json:"tenant_id"`
	UserID    string         `json:"user_id"`
	Query     string         `json:"query"`
	Options   *SearchOptions `json:"options,omitempty"`
}

// SearchResult is a single document match returned by a search.
type SearchResult struct {
	DocumentID string            `json:"document_id"`
	Content    string            `json:"content"`
	Score      float64           `json:"score"`
	Category   string            `json:"category"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// SearchMetadata contains counts and strategy info for a search response.
type SearchMetadata struct {
	Strategy        string `json:"strategy"`
	TotalCandidates int    `json:"total_candidates"`
	FilteredOut     int    `json:"filtered_out"`
	FinalCount      int    `json:"final_count"`
}

// SearchResponse is the result of a search operation.
type SearchResponse struct {
	RequestID        string         `json:"request_id"`
	Results          []SearchResult `json:"results"`
	Metadata         SearchMetadata `json:"metadata"`
	ProcessingTimeMs float64        `json:"processing_time_ms"`
}

// BatchSearchRequest bundles multiple search queries.
type BatchSearchRequest struct {
	RequestID string          `json:"request_id"`
	Queries   []SearchRequest `json:"queries"`
}

// BatchSearchResponse contains one SearchResponse per query.
type BatchSearchResponse struct {
	RequestID string           `json:"request_id"`
	Results   []SearchResponse `json:"results"`
}

// EmbedRequest asks for a single text to be embedded.
type EmbedRequest struct {
	RequestID string `json:"request_id"`
	Text      string `json:"text"`
}

// EmbedResponse returns the embedding vector.
type EmbedResponse struct {
	RequestID string    `json:"request_id"`
	Embedding []float32 `json:"embedding"`
	DimCount  int       `json:"dim_count"`
}

// BatchEmbedRequest asks for multiple texts to be embedded.
type BatchEmbedRequest struct {
	RequestID string   `json:"request_id"`
	Texts     []string `json:"texts"`
}

// BatchEmbedResponse returns one embedding per input text.
type BatchEmbedResponse struct {
	RequestID  string      `json:"request_id"`
	Embeddings [][]float32 `json:"embeddings"`
}

// RerankRequest asks for candidates to be re-scored relative to a query.
type RerankRequest struct {
	RequestID  string         `json:"request_id"`
	Query      string         `json:"query"`
	Candidates []SearchResult `json:"candidates"`
	TopK       int            `json:"top_k"`
}

// RerankResponse returns the re-scored, sorted candidates.
type RerankResponse struct {
	RequestID string         `json:"request_id"`
	Results   []SearchResult `json:"results"`
}

// CollectionInfo describes a single vector collection.
type CollectionInfo struct {
	Name        string `json:"name"`
	VectorCount int    `json:"vector_count"`
	Status      string `json:"status"`
}

// CollectionsRequest is the (empty) payload for listing collections.
type CollectionsRequest struct{}

// CollectionsResponse lists all known collections.
type CollectionsResponse struct {
	Collections []CollectionInfo `json:"collections"`
}

// HealthRequest is the (empty) payload for a health check.
type HealthRequest struct{}

// HealthStatus is returned by health endpoints.
type HealthStatus struct {
	Status  string            `json:"status"`
	Version string            `json:"version"`
	Checks  map[string]string `json:"checks"`
}

// ErrorResponse wraps an error message and HTTP status code.
type ErrorResponse struct {
	Error string `json:"error"`
	Code  int    `json:"code"`
}
