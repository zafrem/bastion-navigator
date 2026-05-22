package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/bastion/navigator/internal/models"
	"github.com/bastion/navigator/internal/orchestrator"
)

// Build constructs the root cobra command tree.
func Build(orch *orchestrator.Orchestrator) *cobra.Command {
	root := &cobra.Command{
		Use:   "navigator-cli",
		Short: "Bastion-Navigator CLI — search, embed, rerank, manage",
		Long: `Bastion-Navigator CLI provides direct access to the search and ranking pipeline.

Run in server mode:
  navigator-cli server --port 8080

Run a single search:
  navigator-cli search --query "warranty terms" --tenant tenant-acme

Interactive shell:
  navigator-cli interactive`,
	}

	root.AddCommand(
		buildSearch(orch),
		buildEmbed(orch),
		buildRerank(orch),
		buildInteractive(orch),
		buildEvaluate(orch),
		buildCollections(orch),
	)
	return root
}

// --- search command ---

func buildSearch(orch *orchestrator.Orchestrator) *cobra.Command {
	var (
		query        string
		tenant       string
		userID       string
		department   string
		topK         int
		overFetch    int
		hybrid       bool
		rerank       bool
		vectorWeight float64
		bm25Weight   float64
		minScore     float64
		filters      []string
		inputFile    string
		outputFile   string
		outputFormat string
		parallel     int
	)

	cmd := &cobra.Command{
		Use:   "search",
		Short: "Execute a search query",
		Example: `  navigator-cli search --query "warranty for PROD-001" --tenant tenant-acme --top-k 10
  navigator-cli search --input-file queries.jsonl --output-file results.jsonl`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := context.Background()

			if inputFile != "" {
				return batchSearchFromFile(ctx, orch, inputFile, outputFile, parallel)
			}
			if query == "" {
				return fmt.Errorf("--query is required (or use --input-file for batch mode)")
			}

			req := buildSearchRequest(query, tenant, userID, department, topK, overFetch,
				hybrid, rerank, vectorWeight, bm25Weight, minScore, filters)

			start := time.Now()
			resp, err := orch.Search(ctx, req)
			if err != nil {
				return err
			}
			resp.ProcessingTimeMs = float64(time.Since(start).Milliseconds())

			printSearchResponse(resp, outputFormat)
			return nil
		},
	}

	cmd.Flags().StringVarP(&query, "query", "q", "", "Search query text")
	cmd.Flags().StringVar(&tenant, "tenant", "", "Tenant ID")
	cmd.Flags().StringVar(&userID, "user-id", "", "User ID for permission filtering")
	cmd.Flags().StringVar(&department, "department", "", "User department")
	cmd.Flags().IntVar(&topK, "top-k", 10, "Number of results to return")
	cmd.Flags().IntVar(&overFetch, "over-fetch", 0, "Initial retrieval size (default: top-k * 5)")
	cmd.Flags().BoolVar(&hybrid, "hybrid", false, "Enable hybrid search (vector + BM25)")
	cmd.Flags().BoolVar(&rerank, "rerank", false, "Enable cross-encoder reranking")
	cmd.Flags().Float64Var(&vectorWeight, "vector-weight", 0.7, "Vector search weight (0-1)")
	cmd.Flags().Float64Var(&bm25Weight, "bm25-weight", 0.3, "BM25 weight (0-1)")
	cmd.Flags().Float64Var(&minScore, "min-score", 0.5, "Minimum relevance score threshold")
	cmd.Flags().StringArrayVar(&filters, "filter", nil, "Metadata filters: key=value")
	cmd.Flags().StringVar(&inputFile, "input-file", "", "JSONL file with batch queries")
	cmd.Flags().StringVar(&outputFile, "output-file", "", "Output file for batch results (stdout if omitted)")
	cmd.Flags().StringVar(&outputFormat, "output-format", "text", "Output format: text|json|compact")
	cmd.Flags().IntVar(&parallel, "parallel", 1, "Parallel batch workers")

	return cmd
}

func buildSearchRequest(
	query, tenant, userID, department string,
	topK, overFetch int,
	hybrid, rerank bool,
	vectorWeight, bm25Weight, minScore float64,
	filters []string,
) models.SearchRequest {
	f := parseFilters(filters)
	req := models.SearchRequest{
		RequestID: fmt.Sprintf("cli-%d", time.Now().UnixNano()),
		TenantID:  tenant,
		Query:     query,
		Options: &models.SearchOptions{
			TopK:         topK,
			OverFetch:    overFetch,
			UseHybrid:    hybrid,
			UseReranking: rerank,
			VectorWeight: vectorWeight,
			BM25Weight:   bm25Weight,
			MinScore:     minScore,
			Filters:      f,
		},
	}
	if userID != "" {
		req.User = &models.UserContext{
			UserID:     userID,
			Department: department,
		}
	}
	return req
}

func parseFilters(raw []string) map[string]string {
	m := make(map[string]string, len(raw))
	for _, kv := range raw {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 2 {
			m[parts[0]] = parts[1]
		}
	}
	return m
}

func batchSearchFromFile(ctx context.Context, orch *orchestrator.Orchestrator, in, out string, _ int) error {
	f, err := os.Open(in)
	if err != nil {
		return err
	}
	defer f.Close()

	var w *os.File
	if out != "" {
		w, err = os.Create(out)
		if err != nil {
			return err
		}
		defer w.Close()
	} else {
		w = os.Stdout
	}

	scanner := bufio.NewScanner(f)
	enc := json.NewEncoder(w)
	i := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var req models.SearchRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			fmt.Fprintf(os.Stderr, "line %d: invalid JSON: %v\n", i+1, err)
			i++
			continue
		}
		resp, err := orch.Search(ctx, req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "line %d: search error: %v\n", i+1, err)
			i++
			continue
		}
		_ = enc.Encode(resp)
		i++
	}
	fmt.Fprintf(os.Stderr, "processed %d queries\n", i)
	return nil
}

// --- embed command ---

func buildEmbed(orch *orchestrator.Orchestrator) *cobra.Command {
	var text, outputFormat string
	cmd := &cobra.Command{
		Use:   "embed",
		Short: "Generate a vector embedding for a text",
		RunE: func(cmd *cobra.Command, args []string) error {
			if text == "" && len(args) > 0 {
				text = strings.Join(args, " ")
			}
			if text == "" {
				return fmt.Errorf("--text is required")
			}
			vec, err := orch.Embed(context.Background(), text)
			if err != nil {
				return err
			}
			if outputFormat == "json" {
				return json.NewEncoder(os.Stdout).Encode(map[string]interface{}{
					"text":      text,
					"embedding": vec,
					"dims":      len(vec),
				})
			}
			fmt.Printf("Text:       %s\nDimensions: %d\nVector[0]:  %.6f\n", text, len(vec), vec[0])
			return nil
		},
	}
	cmd.Flags().StringVar(&text, "text", "", "Text to embed")
	cmd.Flags().StringVar(&outputFormat, "output-format", "text", "text|json")
	return cmd
}

// --- rerank command ---

func buildRerank(orch *orchestrator.Orchestrator) *cobra.Command {
	var query, candidatesFile string
	var topK int
	cmd := &cobra.Command{
		Use:   "rerank",
		Short: "Rerank existing search results with the cross-encoder",
		RunE: func(cmd *cobra.Command, args []string) error {
			if query == "" {
				return fmt.Errorf("--query is required")
			}
			if candidatesFile == "" {
				return fmt.Errorf("--candidates-file is required")
			}
			data, err := os.ReadFile(candidatesFile)
			if err != nil {
				return err
			}
			var candidates []models.SearchResult
			if err := json.Unmarshal(data, &candidates); err != nil {
				return err
			}
			results, err := orch.Rerank(context.Background(), query, candidates, topK)
			if err != nil {
				return err
			}
			return json.NewEncoder(os.Stdout).Encode(results)
		},
	}
	cmd.Flags().StringVarP(&query, "query", "q", "", "Query used for reranking")
	cmd.Flags().StringVar(&candidatesFile, "candidates-file", "", "JSON file with candidate results")
	cmd.Flags().IntVar(&topK, "top-k", 10, "Number of results to return after reranking")
	return cmd
}

// --- collections command ---

func buildCollections(orch *orchestrator.Orchestrator) *cobra.Command {
	return &cobra.Command{
		Use:   "collections",
		Short: "List Qdrant collections",
		RunE: func(cmd *cobra.Command, args []string) error {
			cols, err := orch.Collections(context.Background())
			if err != nil {
				return err
			}
			if len(cols) == 0 {
				fmt.Println("No collections found.")
				return nil
			}
			fmt.Println("Collections:")
			for _, c := range cols {
				fmt.Printf("  - %s (%d vectors, status=%s)\n", c.Name, c.VectorCount, c.Status)
			}
			return nil
		},
	}
}

// --- interactive command ---

func buildInteractive(orch *orchestrator.Orchestrator) *cobra.Command {
	return &cobra.Command{
		Use:   "interactive",
		Short: "Start an interactive search shell",
		RunE: func(cmd *cobra.Command, args []string) error {
			runInteractive(orch)
			return nil
		},
	}
}

func runInteractive(orch *orchestrator.Orchestrator) {
	fmt.Println("Bastion-Navigator interactive mode. Type 'help' for commands, 'exit' to quit.")
	scanner := bufio.NewScanner(os.Stdin)
	searchCount := 0
	var totalLatency time.Duration

	for {
		fmt.Print("nav> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		switch parts[0] {
		case "exit", "quit", "q":
			fmt.Println("Goodbye.")
			return

		case "help":
			fmt.Println(`Commands:
  search <query>      Execute a search
  embed <text>        Generate an embedding
  collections         List collections
  stats               Show session statistics
  exit                Quit`)

		case "search":
			query := strings.Join(parts[1:], " ")
			if query == "" {
				fmt.Print("Query: ")
				if !scanner.Scan() {
					return
				}
				query = scanner.Text()
			}
			start := time.Now()
			resp, err := orch.Search(context.Background(), models.SearchRequest{
				RequestID: fmt.Sprintf("interactive-%d", time.Now().UnixNano()),
				Query:     query,
			})
			elapsed := time.Since(start)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			searchCount++
			totalLatency += elapsed
			fmt.Printf("Found %d results in %dms\n", len(resp.Results), elapsed.Milliseconds())
			for i, r := range resp.Results {
				if i >= 3 {
					fmt.Printf("  ... and %d more\n", len(resp.Results)-3)
					break
				}
				content := r.Content
				if len(content) > 80 {
					content = content[:80] + "..."
				}
				fmt.Printf("  #%d [%.3f] %s: %s\n", i+1, r.Score, r.DocumentID, content)
			}

		case "embed":
			text := strings.Join(parts[1:], " ")
			if text == "" {
				fmt.Println("Usage: embed <text>")
				continue
			}
			vec, err := orch.Embed(context.Background(), text)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			fmt.Printf("Embedding: %d dimensions, first value: %.6f\n", len(vec), vec[0])

		case "collections":
			cols, err := orch.Collections(context.Background())
			if err != nil {
				fmt.Printf("Error: %v\n", err)
				continue
			}
			for _, c := range cols {
				fmt.Printf("  - %s (%d vectors)\n", c.Name, c.VectorCount)
			}

		case "stats":
			avg := time.Duration(0)
			if searchCount > 0 {
				avg = totalLatency / time.Duration(searchCount)
			}
			fmt.Printf("Searches:    %d\nAvg latency: %dms\n", searchCount, avg.Milliseconds())

		default:
			fmt.Printf("Unknown command: %s (type 'help')\n", parts[0])
		}
	}
}

// --- evaluate command ---

func buildEvaluate(orch *orchestrator.Orchestrator) *cobra.Command {
	var queriesFile, groundTruthFile, metricsFlag string
	cmd := &cobra.Command{
		Use:   "evaluate",
		Short: "Run quality benchmark against ground truth",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runEvaluate(orch, queriesFile, groundTruthFile, metricsFlag)
		},
	}
	cmd.Flags().StringVar(&queriesFile, "queries", "", "JSONL file with queries")
	cmd.Flags().StringVar(&groundTruthFile, "ground-truth", "", "JSONL file with ground truth")
	cmd.Flags().StringVar(&metricsFlag, "metrics", "recall,precision,ndcg,mrr", "Comma-separated metrics")
	_ = cmd.MarkFlagRequired("queries")
	_ = cmd.MarkFlagRequired("ground-truth")
	return cmd
}

type groundTruth struct {
	QueryID    string   `json:"query_id"`
	RelevantIDs []string `json:"relevant_ids"`
}

func runEvaluate(orch *orchestrator.Orchestrator, queriesFile, gtFile, _ string) error {
	queries, err := loadJSONL[models.SearchRequest](queriesFile)
	if err != nil {
		return fmt.Errorf("load queries: %w", err)
	}
	gts, err := loadJSONL[groundTruth](gtFile)
	if err != nil {
		return fmt.Errorf("load ground truth: %w", err)
	}
	gtMap := make(map[string]map[string]struct{}, len(gts))
	for _, g := range gts {
		set := make(map[string]struct{}, len(g.RelevantIDs))
		for _, id := range g.RelevantIDs {
			set[id] = struct{}{}
		}
		gtMap[g.QueryID] = set
	}

	var totalRecall, totalPrecision, totalNDCG, totalMRR float64
	n := 0
	for _, q := range queries {
		resp, err := orch.Search(context.Background(), q)
		if err != nil {
			continue
		}
		relevant, ok := gtMap[q.RequestID]
		if !ok {
			continue
		}
		k := len(resp.Results)
		if k == 0 {
			n++
			continue
		}
		hits := 0
		var rr float64
		var dcg float64
		for i, r := range resp.Results {
			if _, isRel := relevant[r.DocumentID]; isRel {
				hits++
				if rr == 0 {
					rr = 1.0 / float64(i+1)
				}
				gain := 1.0 / log2(float64(i+2))
				dcg += gain
			}
		}
		precision := float64(hits) / float64(k)
		recall := float64(hits) / float64(len(relevant))
		idealDCG := idealDCGAt(len(relevant), k)
		ndcg := 0.0
		if idealDCG > 0 {
			ndcg = dcg / idealDCG
		}
		totalPrecision += precision
		totalRecall += recall
		totalNDCG += ndcg
		totalMRR += rr
		n++
	}

	if n == 0 {
		fmt.Println("No queries could be evaluated.")
		return nil
	}
	fmt.Printf("\nQuality Report:\n")
	fmt.Printf("─────────────────────────\n")
	fmt.Printf("Recall@K:    %.2f\n", totalRecall/float64(n))
	fmt.Printf("Precision@K: %.2f\n", totalPrecision/float64(n))
	fmt.Printf("NDCG@K:      %.2f\n", totalNDCG/float64(n))
	fmt.Printf("MRR:         %.2f\n", totalMRR/float64(n))
	fmt.Printf("─────────────────────────\n")
	return nil
}

func idealDCGAt(numRelevant, k int) float64 {
	dcg := 0.0
	for i := 0; i < k && i < numRelevant; i++ {
		dcg += 1.0 / log2(float64(i+2))
	}
	return dcg
}

func log2(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// log2(x) = ln(x)/ln(2)
	result := 0.0
	for x > 1 {
		x /= 2
		result++
	}
	return result
}

func loadJSONL[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []T
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var item T
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, nil
}

// --- output formatting ---

func printSearchResponse(resp models.SearchResponse, format string) {
	switch format {
	case "json":
		_ = json.NewEncoder(os.Stdout).Encode(resp)
	case "compact":
		fmt.Printf("[%s] search results=%d time=%.0fms strategy=%s\n",
			resp.RequestID, resp.Metadata.FinalCount,
			resp.ProcessingTimeMs, resp.Metadata.Strategy)
	default:
		printTextResponse(resp)
	}
}

func printTextResponse(resp models.SearchResponse) {
	sep := strings.Repeat("═", 56)
	thin := strings.Repeat("─", 56)

	fmt.Println(sep)
	fmt.Println("  Bastion-Navigator Search Result")
	fmt.Println(sep)
	fmt.Printf("Request ID:   %s\n", resp.RequestID)
	fmt.Printf("Strategy:     %s\n", resp.Metadata.Strategy)
	fmt.Printf("Time:         %.1f ms\n\n", resp.ProcessingTimeMs)

	fmt.Printf("─── Search Metadata %s\n", thin[19:])
	fmt.Printf("Initial candidates:  %d\n", resp.Metadata.TotalCandidates)
	fmt.Printf("Permission filtered: %d\n", resp.Metadata.FilteredOut)
	fmt.Printf("Final results:       %d\n", resp.Metadata.FinalCount)
	fmt.Printf("Cache hit:           %v\n\n", resp.Metadata.UsedCache)

	fmt.Printf("─── Top Results %s\n", thin[15:])
	for i, r := range resp.Results {
		content := r.Content
		if len(content) > 120 {
			content = content[:120] + "..."
		}
		fmt.Printf("\n#%d  [score: %.2f] %s  (%s)\n", i+1, r.Score, r.DocumentID, r.Category)
		fmt.Printf("    Vector: %.2f | BM25: %.2f | Rerank: %.2f\n",
			r.VectorScore, r.BM25Score, r.RerankScore)
		if content != "" {
			fmt.Printf("    %q\n", content)
		}
		if len(r.Metadata) > 0 {
			parts := make([]string, 0, len(r.Metadata))
			for k, v := range r.Metadata {
				parts = append(parts, k+"="+v)
			}
			fmt.Printf("    Metadata: %s\n", strings.Join(parts, ", "))
		}
	}
	fmt.Println("\n" + sep)
}
