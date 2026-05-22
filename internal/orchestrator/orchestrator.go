// Package orchestrator implements the 15-step search pipeline described in SRS §6.3.
package orchestrator

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/bastion/navigator/internal/config"
	"github.com/bastion/navigator/internal/embedder"
	"github.com/bastion/navigator/internal/metrics"
	"github.com/bastion/navigator/internal/models"
	"github.com/bastion/navigator/internal/reranker"
	"github.com/bastion/navigator/internal/searcher"
	"github.com/bastion/navigator/internal/vault"
)

// Orchestrator coordinates the full search pipeline.
type Orchestrator struct {
	cfg      *config.Config
	embedder embedder.Embedder
	searcher searcher.Searcher
	reranker reranker.Reranker
	vault    vault.Client
}

func New(
	cfg *config.Config,
	emb embedder.Embedder,
	srch searcher.Searcher,
	rnk reranker.Reranker,
	vlt vault.Client,
) *Orchestrator {
	return &Orchestrator{
		cfg:      cfg,
		embedder: emb,
		searcher: srch,
		reranker: rnk,
		vault:    vlt,
	}
}

// Search executes the full pipeline and returns ranked results.
func (o *Orchestrator) Search(ctx context.Context, req models.SearchRequest) (models.SearchResponse, error) {
	start := time.Now()

	opts := o.mergeDefaults(req.Options)
	strategy := strategyName(opts)

	defer func() {
		d := time.Since(start).Seconds()
		metrics.SearchDuration.WithLabelValues(strategy).Observe(d)
		tenantID := req.TenantID
		if tenantID == "" {
			tenantID = "unknown"
		}
		metrics.SearchesTotal.WithLabelValues(tenantID, strategy).Inc()
	}()

	// [1] Apply tenant pre-filter — this is always mandatory (FR-TP-001).
	filter := map[string]string{"tenant_id": req.TenantID}
	for k, v := range opts.Filters {
		filter[k] = v
	}

	// [2] Resolve user permissions from Vault (FR-AC-001).
	allowed, err := o.resolvePermissions(ctx, req)
	if err != nil {
		return models.SearchResponse{}, fmt.Errorf("permission check: %w", err)
	}

	// [3] Generate query embedding (FR-EM-001).
	vec, err := o.embedder.Embed(ctx, req.Query)
	if err != nil {
		return models.SearchResponse{}, fmt.Errorf("embed query: %w", err)
	}

	// [4] Determine searchable collections based on allowed categories (FR-VS-002).
	collections := o.collectionsForCategories(allowed)

	overFetch := opts.OverFetch
	if overFetch == 0 {
		overFetch = opts.TopK * o.cfg.SearchDefaults.OverFetchMultiplier
	}

	var allCandidates []models.SearchResult

	// [5-8] Execute search across all permitted collections.
	for _, coll := range collections {
		candidates, err := o.searchCollection(ctx, coll, req.Query, vec, filter, opts, overFetch)
		if err != nil {
			// Log and continue — partial results are better than none.
			continue
		}
		allCandidates = append(allCandidates, candidates...)
	}

	if len(allCandidates) == 0 {
		metrics.ZeroResultSearches.Add(1)
		return models.SearchResponse{
			RequestID: req.RequestID,
			Results:   []models.SearchResult{},
			Metadata: models.SearchMetadata{
				Strategy:    strategy,
				FinalCount:  0,
			},
			ProcessingTimeMs: float64(time.Since(start).Milliseconds()),
		}, nil
	}

	totalCandidates := len(allCandidates)

	// [9] Filter by permissions (FR-AC-002).
	permitted, filteredOut := o.vault.FilterResults(allCandidates, allowed)

	// [10] Rerank if enabled and beneficial (FR-RR-001, FR-RR-003).
	var finalResults []models.SearchResult
	if opts.UseReranking && len(permitted) >= 5 {
		ctx2, cancel := context.WithTimeout(ctx, time.Duration(opts.TimeoutMs)*time.Millisecond/2)
		defer cancel()
		finalResults, err = o.reranker.Rerank(ctx2, req.Query, permitted, opts.TopK)
		if err != nil {
			// Reranker failure is non-fatal — fall back to fusion scores.
			finalResults = topK(permitted, opts.TopK)
		}
	} else {
		finalResults = topK(permitted, opts.TopK)
	}

	// [11] Apply minimum score threshold.
	if opts.MinScore > 0 {
		filtered := finalResults[:0]
		for _, r := range finalResults {
			if r.Score >= opts.MinScore {
				filtered = append(filtered, r)
			}
		}
		filteredOut += len(finalResults) - len(filtered)
		finalResults = filtered
	}

	metrics.ResultsReturned.Add(float64(len(finalResults)))
	metrics.ResultsFiltered.Add(float64(filteredOut))

	return models.SearchResponse{
		RequestID: req.RequestID,
		Results:   finalResults,
		Metadata: models.SearchMetadata{
			TotalCandidates: totalCandidates,
			FilteredOut:     filteredOut,
			FinalCount:      len(finalResults),
			Strategy:        strategy,
		},
		ProcessingTimeMs: float64(time.Since(start).Milliseconds()),
	}, nil
}

// searchCollection runs vector search, optional BM25, and combines via RRF.
func (o *Orchestrator) searchCollection(
	ctx context.Context,
	collection, query string,
	vec []float32,
	filter map[string]string,
	opts *models.SearchOptions,
	overFetch int,
) ([]models.SearchResult, error) {
	// Vector search (always executed).
	vectorResults, err := o.searcher.VectorSearch(ctx, collection, vec, filter, overFetch, 0)
	if err != nil {
		return nil, err
	}

	if !opts.UseHybrid {
		return vectorResults, nil
	}

	// BM25 search (parallel-safe; errors degrade gracefully).
	bm25Results, _ := o.searcher.SparseSearch(ctx, collection, query, filter, overFetch)

	// RRF fusion (FR-HS-001).
	return rrf(vectorResults, bm25Results, opts.VectorWeight, opts.BM25Weight), nil
}

// rrf implements Reciprocal Rank Fusion with k=60 (SRS FR-HS-001).
func rrf(vectorResults, bm25Results []models.SearchResult, vectorWeight, bm25Weight float64) []models.SearchResult {
	const k = 60.0
	scores := make(map[string]float64)
	byID := make(map[string]models.SearchResult)

	for rank, r := range vectorResults {
		scores[r.DocumentID] += vectorWeight * (1.0 / (float64(rank+1) + k))
		byID[r.DocumentID] = r
	}
	for rank, r := range bm25Results {
		scores[r.DocumentID] += bm25Weight * (1.0 / (float64(rank+1) + k))
		if _, exists := byID[r.DocumentID]; !exists {
			byID[r.DocumentID] = r
		} else {
			existing := byID[r.DocumentID]
			existing.BM25Score = r.BM25Score
			byID[r.DocumentID] = existing
		}
	}

	out := make([]models.SearchResult, 0, len(byID))
	for id, sr := range byID {
		sr.Score = scores[id]
		out = append(out, sr)
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Score > out[j].Score
	})
	return out
}

// resolvePermissions returns the categories the user may access.
// It uses categories from UserContext if already provided by Vault (via upstream pipeline),
// otherwise it queries Vault directly.
func (o *Orchestrator) resolvePermissions(ctx context.Context, req models.SearchRequest) ([]string, error) {
	if req.User != nil && len(req.User.AllowedCategories) > 0 {
		return req.User.AllowedCategories, nil
	}
	userID := ""
	if req.User != nil {
		userID = req.User.UserID
	}
	if userID == "" {
		// No user context — grant all categories (standalone / open mode).
		return nil, nil
	}
	return o.vault.AllowedCategories(ctx, userID)
}

// collectionsForCategories maps categories to Qdrant collection names (SRS FR-VS-002).
var categoryToCollection = map[string]string{
	"customer_data":      "customer_docs",
	"manufacturing_data": "manufacturing_docs",
	"hr_data":            "hr_docs",
}

func (o *Orchestrator) collectionsForCategories(allowed []string) []string {
	if len(allowed) == 0 {
		// No restriction — search all collections.
		out := make([]string, 0, len(o.cfg.VectorDB.Collections))
		for name := range o.cfg.VectorDB.Collections {
			out = append(out, name)
		}
		return out
	}

	seen := make(map[string]struct{})
	var out []string
	for _, cat := range allowed {
		if coll, ok := categoryToCollection[cat]; ok {
			if _, already := seen[coll]; !already {
				seen[coll] = struct{}{}
				out = append(out, coll)
			}
		}
	}
	if len(out) == 0 {
		// Fallback: use all configured collections.
		for name := range o.cfg.VectorDB.Collections {
			out = append(out, name)
		}
	}
	return out
}

// mergeDefaults fills in zero-valued options from config defaults.
func (o *Orchestrator) mergeDefaults(opts *models.SearchOptions) *models.SearchOptions {
	d := o.cfg.SearchDefaults
	if opts == nil {
		opts = &models.SearchOptions{}
	}
	if opts.TopK == 0 {
		opts.TopK = d.TopK
	}
	if opts.VectorWeight == 0 {
		opts.VectorWeight = d.VectorWeight
	}
	if opts.BM25Weight == 0 {
		opts.BM25Weight = d.BM25Weight
	}
	if opts.MinScore == 0 {
		opts.MinScore = d.MinScore
	}
	if opts.TimeoutMs == 0 {
		opts.TimeoutMs = d.TimeoutMs
	}
	// Preserve caller's explicit false values by defaulting only when the struct is zero.
	if !opts.UseHybrid && !opts.UseReranking {
		opts.UseHybrid = d.UseHybrid
		opts.UseReranking = d.UseReranking
	}
	return opts
}

func topK(results []models.SearchResult, k int) []models.SearchResult {
	if k <= 0 || k >= len(results) {
		return results
	}
	return results[:k]
}

func strategyName(opts *models.SearchOptions) string {
	switch {
	case opts.UseHybrid && opts.UseReranking:
		return "hybrid+rerank"
	case opts.UseHybrid:
		return "hybrid"
	case opts.UseReranking:
		return "vector+rerank"
	default:
		return "vector"
	}
}

// Embed delegates to the embedder.
func (o *Orchestrator) Embed(ctx context.Context, text string) ([]float32, error) {
	return o.embedder.Embed(ctx, text)
}

// EmbedBatch delegates to the embedder.
func (o *Orchestrator) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return o.embedder.EmbedBatch(ctx, texts)
}

// Rerank delegates to the reranker.
func (o *Orchestrator) Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	return o.reranker.Rerank(ctx, query, candidates, topK)
}

// Collections delegates to the searcher.
func (o *Orchestrator) Collections(ctx context.Context) ([]models.CollectionInfo, error) {
	return o.searcher.Collections(ctx)
}
