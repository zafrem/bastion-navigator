// Package orchestrator wires together embedding, search, reranking, and
// permission-filtering into a single Search call.
package orchestrator

import (
	"context"
	"log"
	"sort"
	"time"

	"github.com/zafrem/bastion-navigator/internal/config"
	"github.com/zafrem/bastion-navigator/internal/models"
)

// Embedder produces dense vector representations of text.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// Searcher queries a vector store.
type Searcher interface {
	VectorSearch(ctx context.Context, collection string, vec []float32, filters map[string]string, limit int, minScore float64) ([]models.SearchResult, error)
	SparseSearch(ctx context.Context, collection string, text string, filters map[string]string, limit int) ([]models.SearchResult, error)
	Collections(ctx context.Context) ([]models.CollectionInfo, error)
}

// Reranker re-scores search results relative to a query.
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error)
}

// Vault resolves a user's permitted data categories and filters results to them. A nil category slice
// means unrestricted (Vault disabled); a non-nil empty one means no access.
type Vault interface {
	AllowedCategories(ctx context.Context, user models.UserContext) ([]string, error)
	FilterResults(results []models.SearchResult, categories []string) ([]models.SearchResult, int)
}

// Orchestrator coordinates the full search pipeline.
type Orchestrator struct {
	cfg      *config.Config
	embedder Embedder
	searcher Searcher
	reranker Reranker
	vault    Vault
}

// New returns an Orchestrator wired with the supplied backends.
func New(cfg *config.Config, emb Embedder, srch Searcher, rnk Reranker, vlt Vault) *Orchestrator {
	return &Orchestrator{cfg: cfg, embedder: emb, searcher: srch, reranker: rnk, vault: vlt}
}

// Search runs the full pipeline: embed → vector/hybrid search → permission filter → (re-)rank.
func (o *Orchestrator) Search(ctx context.Context, req models.SearchRequest) (models.SearchResponse, error) {
	start := time.Now()
	opts := o.mergeDefaults(req.Options)

	// Resolve permissions before doing any work: no access means no search.
	allowed := o.resolvePermissions(ctx, req)
	if allowed != nil && len(allowed) == 0 {
		return models.SearchResponse{
			RequestID:        req.RequestID,
			Results:          []models.SearchResult{},
			Metadata:         models.SearchMetadata{Strategy: "denied"},
			ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000.0,
		}, nil
	}

	vec, err := o.embedder.Embed(ctx, req.Query)
	if err != nil {
		return models.SearchResponse{}, err
	}

	filters := map[string]string{"tenant_id": req.TenantID}
	if opts.Filters != nil {
		for k, v := range opts.Filters {
			filters[k] = v
		}
	}

	topK := opts.TopK
	if topK <= 0 {
		topK = 10
	}
	overFetch := topK * o.cfg.SearchDefaults.OverFetchMultiplier
	if overFetch <= 0 {
		overFetch = topK * 5
	}

	cols, _ := o.searcher.Collections(ctx)

	var all []models.SearchResult
	for _, col := range cols {
		res, err := o.searcher.VectorSearch(ctx, col.Name, vec, filters, overFetch, opts.MinScore)
		if err == nil {
			all = append(all, res...)
		}
	}

	if opts.UseHybrid {
		for _, col := range cols {
			sparse, err := o.searcher.SparseSearch(ctx, col.Name, req.Query, filters, overFetch)
			if err == nil {
				all = append(all, sparse...)
			}
		}
		all = dedup(all)
	}

	total := len(all)
	filtered, filteredOut := o.vault.FilterResults(all, allowed)

	strategy := "vector"
	if opts.UseHybrid && opts.UseReranking {
		strategy = "hybrid_rerank"
	} else if opts.UseHybrid {
		strategy = "hybrid"
	} else if opts.UseReranking {
		strategy = "rerank"
	}

	if opts.UseReranking && len(filtered) > 0 {
		reranked, err := o.reranker.Rerank(ctx, req.Query, filtered, topK)
		if err == nil {
			filtered = reranked
		} else {
			filtered = topKSlice(filtered, topK)
		}
	} else {
		filtered = topKSlice(filtered, topK)
	}

	if opts.MinScore > 0 {
		var pass []models.SearchResult
		for _, r := range filtered {
			if r.Score >= opts.MinScore {
				pass = append(pass, r)
			}
		}
		filtered = pass
	}

	dur := float64(time.Since(start).Microseconds()) / 1000.0
	return models.SearchResponse{
		RequestID: req.RequestID,
		Results:   filtered,
		Metadata: models.SearchMetadata{
			Strategy:        strategy,
			TotalCandidates: total,
			FilteredOut:     filteredOut,
			FinalCount:      len(filtered),
		},
		ProcessingTimeMs: dur,
	}, nil
}

// resolvePermissions returns the categories the request's user may search: nil is unrestricted (Vault
// disabled), an empty slice is no access. Vault is authoritative; categories supplied by the caller can
// only narrow its answer. If Vault cannot be consulted, access is denied rather than granted.
func (o *Orchestrator) resolvePermissions(ctx context.Context, req models.SearchRequest) []string {
	allowed, err := o.vault.AllowedCategories(ctx, models.UserContext{
		UserID:     req.UserID,
		TenantID:   req.TenantID,
		Department: req.Department,
		Roles:      req.Roles,
	})
	if err != nil {
		log.Printf("[orchestrator] Vault permission lookup failed for %q, denying access: %v", req.UserID, err)
		return []string{}
	}
	if allowed == nil {
		if len(req.AllowedCategories) > 0 {
			return req.AllowedCategories
		}
		return nil
	}
	if len(req.AllowedCategories) == 0 {
		return allowed
	}
	permitted := make(map[string]struct{}, len(allowed))
	for _, c := range allowed {
		permitted[c] = struct{}{}
	}
	narrowed := []string{}
	for _, c := range req.AllowedCategories {
		if _, ok := permitted[c]; ok {
			narrowed = append(narrowed, c)
		}
	}
	return narrowed
}

// Embed delegates to the underlying embedder.
func (o *Orchestrator) Embed(ctx context.Context, text string) ([]float32, error) {
	return o.embedder.Embed(ctx, text)
}

// EmbedBatch delegates to the underlying embedder.
func (o *Orchestrator) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return o.embedder.EmbedBatch(ctx, texts)
}

// Rerank delegates to the underlying reranker.
func (o *Orchestrator) Rerank(ctx context.Context, query string, candidates []models.SearchResult, topK int) ([]models.SearchResult, error) {
	return o.reranker.Rerank(ctx, query, candidates, topK)
}

// Collections returns all known vector collections from the searcher.
func (o *Orchestrator) Collections(ctx context.Context) ([]models.CollectionInfo, error) {
	return o.searcher.Collections(ctx)
}

func (o *Orchestrator) mergeDefaults(opts *models.SearchOptions) models.SearchOptions {
	d := o.cfg.SearchDefaults
	result := models.SearchOptions{
		TopK:         d.TopK,
		UseHybrid:    d.UseHybrid,
		UseReranking: d.UseReranking,
		MinScore:     d.MinScore,
	}
	if opts == nil {
		return result
	}
	if opts.TopK > 0 {
		result.TopK = opts.TopK
	}
	result.UseHybrid = opts.UseHybrid
	result.UseReranking = opts.UseReranking
	if opts.MinScore > 0 {
		result.MinScore = opts.MinScore
	}
	if opts.Filters != nil {
		result.Filters = opts.Filters
	}
	return result
}

func topKSlice(results []models.SearchResult, k int) []models.SearchResult {
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if k > 0 && k < len(results) {
		return results[:k]
	}
	return results
}

func dedup(results []models.SearchResult) []models.SearchResult {
	seen := make(map[string]struct{}, len(results))
	out := results[:0]
	for _, r := range results {
		if _, ok := seen[r.DocumentID]; !ok {
			seen[r.DocumentID] = struct{}{}
			out = append(out, r)
		}
	}
	return out
}
