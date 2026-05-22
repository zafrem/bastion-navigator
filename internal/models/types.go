package models

// SearchRequest is the input for a search operation.
type SearchRequest struct {
	RequestID string        `json:"request_id"`
	TenantID  string        `json:"tenant_id"`
	Query     string        `json:"query"`
	User      *UserContext  `json:"user,omitempty"`
	Options   *SearchOptions `json:"options,omitempty"`
}

// UserContext carries caller identity and pre-resolved permissions from Vault.
type UserContext struct {
	UserID            string   `json:"user_id"`
	Department        string   `json:"department"`
	Roles             []string `json:"roles"`
	AllowedCategories []string `json:"allowed_categories"`
}

// SearchOptions tunes a single search call.
type SearchOptions struct {
	TopK          int               `json:"top_k"`
	OverFetch     int               `json:"over_fetch"`
	UseReranking  bool              `json:"use_reranking"`
	UseHybrid     bool              `json:"use_hybrid"`
	VectorWeight  float64           `json:"vector_weight"`
	BM25Weight    float64           `json:"bm25_weight"`
	Filters       map[string]string `json:"filters"`
	MinScore      float64           `json:"min_score"`
	TimeoutMs     int               `json:"timeout_ms"`
}

// SearchResponse is the result of a search operation.
type SearchResponse struct {
	RequestID         string         `json:"request_id"`
	Results           []SearchResult `json:"results"`
	Metadata          SearchMetadata `json:"metadata"`
	ProcessingTimeMs  float64        `json:"processing_time_ms"`
}

// SearchResult is a single ranked document.
type SearchResult struct {
	DocumentID  string            `json:"document_id"`
	Content     string            `json:"content"`
	Score       float64           `json:"score"`
	VectorScore float64           `json:"vector_score"`
	BM25Score   float64           `json:"bm25_score"`
	RerankScore float64           `json:"rerank_score"`
	Metadata    map[string]string `json:"metadata"`
	Category    string            `json:"category"`
}

// SearchMetadata carries pipeline statistics.
type SearchMetadata struct {
	TotalCandidates int    `json:"total_candidates"`
	FilteredOut     int    `json:"filtered_out"`
	FinalCount      int    `json:"final_count"`
	UsedCache       bool   `json:"used_cache"`
	Strategy        string `json:"strategy"`
}

// EmbedRequest asks for the vector representation of a text.
type EmbedRequest struct {
	RequestID string `json:"request_id"`
	Text      string `json:"text"`
}

// EmbedResponse carries the resulting vector.
type EmbedResponse struct {
	RequestID string    `json:"request_id"`
	Embedding []float32 `json:"embedding"`
	DimCount  int       `json:"dim_count"`
}

// BatchEmbedRequest batches multiple texts.
type BatchEmbedRequest struct {
	RequestID string   `json:"request_id"`
	Texts     []string `json:"texts"`
}

// BatchEmbedResponse carries embeddings for each text.
type BatchEmbedResponse struct {
	RequestID  string      `json:"request_id"`
	Embeddings [][]float32 `json:"embeddings"`
}

// RerankCandidate is a document to be re-scored.
type RerankCandidate struct {
	DocumentID string `json:"document_id"`
	Content    string `json:"content"`
}

// RerankRequest asks the cross-encoder to re-score candidates.
type RerankRequest struct {
	RequestID  string            `json:"request_id"`
	Query      string            `json:"query"`
	Candidates []SearchResult    `json:"candidates"`
	TopK       int               `json:"top_k"`
}

// RerankResponse carries re-ordered results.
type RerankResponse struct {
	RequestID string         `json:"request_id"`
	Results   []SearchResult `json:"results"`
}

// BatchSearchRequest groups multiple search queries.
type BatchSearchRequest struct {
	RequestID string          `json:"request_id"`
	Queries   []SearchRequest `json:"queries"`
}

// BatchSearchResponse groups multiple search results.
type BatchSearchResponse struct {
	RequestID string           `json:"request_id"`
	Results   []SearchResponse `json:"results"`
}

// CollectionInfo describes a Qdrant collection.
type CollectionInfo struct {
	Name        string `json:"name"`
	VectorCount int64  `json:"vector_count"`
	Dimensions  int    `json:"dimensions"`
	Status      string `json:"status"`
}

// CollectionsResponse lists all collections.
type CollectionsResponse struct {
	Collections []CollectionInfo `json:"collections"`
}

// HealthStatus represents service health.
type HealthStatus struct {
	Status    string            `json:"status"`
	Checks    map[string]string `json:"checks"`
	Version   string            `json:"version"`
}

// ErrorResponse is the standard error envelope.
type ErrorResponse struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	TraceID string `json:"trace_id,omitempty"`
}

// CollectionsRequest is the empty request for listing collections.
type CollectionsRequest struct{}

// HealthRequest is the empty request for the health RPC.
type HealthRequest struct{}
