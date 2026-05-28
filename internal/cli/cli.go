// Package cli provides navigator-cli sub-commands for interactive use.
package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/bastion/navigator/internal/models"
	"github.com/bastion/navigator/internal/orchestrator"
)

// Build returns a *cobra.Command whose sub-commands use orch to search,
// embed, rerank, and inspect collections.
func Build(orch *orchestrator.Orchestrator) *cobra.Command {
	root := &cobra.Command{
		Use:   "navigator",
		Short: "Bastion Navigator CLI — RAG search and ranking",
		Long:  "Interact with Navigator: search documents, inspect embeddings, evaluate relevance.",
	}

	root.AddCommand(
		buildSearchCmd(orch),
		buildEmbedCmd(orch),
		buildCollectionsCmd(orch),
	)
	return root
}

func buildSearchCmd(orch *orchestrator.Orchestrator) *cobra.Command {
	var tenantID string
	var hybrid bool
	var topK int

	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Search for documents matching a query",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.Join(args, " ")
			req := models.SearchRequest{
				TenantID: tenantID,
				Query:    query,
				Options: &models.SearchOptions{
					UseHybrid: hybrid,
					TopK:      topK,
				},
			}
			resp, err := orch.Search(context.Background(), req)
			if err != nil {
				return fmt.Errorf("search: %w", err)
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(resp)
		},
	}
	cmd.Flags().StringVar(&tenantID, "tenant", "", "Tenant ID for permission filtering")
	cmd.Flags().BoolVar(&hybrid, "hybrid", false, "Use hybrid (vector+sparse) search")
	cmd.Flags().IntVar(&topK, "top-k", 10, "Number of results to return")
	return cmd
}

func buildEmbedCmd(orch *orchestrator.Orchestrator) *cobra.Command {
	return &cobra.Command{
		Use:   "embed <text>",
		Short: "Print the embedding vector for the given text",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			text := strings.Join(args, " ")
			vec, err := orch.Embed(context.Background(), text)
			if err != nil {
				return fmt.Errorf("embed: %w", err)
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(map[string]interface{}{
				"dim_count": len(vec),
				"embedding": vec,
			})
		},
	}
}

func buildCollectionsCmd(orch *orchestrator.Orchestrator) *cobra.Command {
	return &cobra.Command{
		Use:   "collections",
		Short: "List available vector collections",
		RunE: func(cmd *cobra.Command, args []string) error {
			cols, err := orch.Collections(context.Background())
			if err != nil {
				return fmt.Errorf("collections: %w", err)
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(cols)
		},
	}
}
