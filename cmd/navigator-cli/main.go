package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	grpcserver "github.com/bastion/navigator/internal/api/grpc"
	"github.com/bastion/navigator/internal/api/rest"
	"github.com/bastion/navigator/internal/cache"
	"github.com/bastion/navigator/internal/cli"
	"github.com/bastion/navigator/internal/config"
	"github.com/bastion/navigator/internal/embedder"
	"github.com/bastion/navigator/internal/orchestrator"
	"github.com/bastion/navigator/internal/reranker"
	"github.com/bastion/navigator/internal/searcher"
	"github.com/bastion/navigator/internal/vault"
)

var (
	cfgPath    string
	standalone bool
)

func main() {
	root := &cobra.Command{
		Use:   "navigator-cli",
		Short: "Bastion-Navigator — RAG search and ranking layer",
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			return nil // config is loaded lazily per sub-command
		},
	}

	root.PersistentFlags().StringVar(&cfgPath, "config", "/etc/bastion-navigator/config.yaml", "Config file path")
	root.PersistentFlags().BoolVar(&standalone, "standalone", false, "Standalone mode: mock Vault and use random embeddings")

	// server sub-command (starts REST + gRPC listeners)
	var restPort, grpcPort int
	serverCmd := &cobra.Command{
		Use:   "server",
		Short: "Start the Navigator API server",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServer(restPort, grpcPort)
		},
	}
	serverCmd.Flags().IntVar(&restPort, "port", 8080, "REST API port")
	serverCmd.Flags().IntVar(&grpcPort, "grpc-port", 9090, "gRPC port")

	root.AddCommand(serverCmd)

	// Attach search/embed/rerank/interactive/evaluate commands.
	// These require building the orchestrator.
	orch, err := buildOrchestrator()
	if err != nil {
		// Not all environments have the full stack; print warning but don't exit.
		fmt.Fprintf(os.Stderr, "⚠️  Could not connect to backend services: %v\n", err)
		fmt.Fprintf(os.Stderr, "   Use --standalone to run without external dependencies.\n")
		// Still register commands so --help works.
		orch = buildStandaloneOrchestrator()
	}

	for _, sub := range cli.Build(orch).Commands() {
		root.AddCommand(sub)
	}
	// Also add the top-level Use/Short commands from cli.Build.
	cliRoot := cli.Build(orch)
	root.Short = cliRoot.Short
	root.Long = cliRoot.Long

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

func runServer(restPort, grpcPort int) error {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Printf("config load failed (%v), using defaults", err)
		cfg = config.Defaults()
	}
	if restPort != 0 {
		cfg.Server.RESTPort = restPort
	}
	if grpcPort != 0 {
		cfg.Server.GRPCPort = grpcPort
	}

	fmt.Println("🚀 Bastion-Navigator v1.0 starting...")

	orch, err := buildOrchestratorFromConfig(cfg)
	if err != nil {
		if !standalone {
			return fmt.Errorf("init orchestrator: %w", err)
		}
		fmt.Fprintln(os.Stderr, "⚠️  Backend unavailable, running in standalone mode")
		orch = buildStandaloneOrchestrator()
	}
	fmt.Printf("✅ REST API on :%d\n", cfg.Server.RESTPort)
	fmt.Printf("✅ gRPC API on :%d\n", cfg.Server.GRPCPort)
	fmt.Println("✨ Ready")

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Run REST and gRPC servers concurrently; either failing stops both.
	restSrv := rest.New(orch, cfg.Server.RESTPort)
	grpcSrv := grpcserver.New(orch, cfg.Server.GRPCPort)

	errCh := make(chan error, 2)
	go func() { errCh <- restSrv.Start(ctx) }()
	go func() { errCh <- grpcSrv.Start(ctx) }()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return nil
	}
}

func buildOrchestrator() (*orchestrator.Orchestrator, error) {
	if standalone {
		return buildStandaloneOrchestrator(), nil
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		cfg = config.Defaults()
	}
	return buildOrchestratorFromConfig(cfg)
}

func buildOrchestratorFromConfig(cfg *config.Config) (*orchestrator.Orchestrator, error) {
	// Cache
	var c cache.Cache
	if cfg.Cache.URL != "" {
		rc, err := cache.NewRedis(cfg.Cache.URL)
		if err != nil {
			log.Printf("Redis unavailable (%v), falling back to in-memory cache", err)
			c = cache.NewInMemory()
		} else {
			c = rc
		}
	} else {
		c = cache.NewInMemory()
	}

	// Embedder
	var emb embedder.Embedder
	if cfg.Embedder.Endpoint != "" {
		emb = embedder.NewBGE(cfg.Embedder, c)
	} else {
		emb = embedder.NewMock(cfg.VectorDB.Collections["customer_docs"].VectorSize)
	}

	// Searcher
	if len(cfg.VectorDB.Hosts) == 0 {
		return nil, fmt.Errorf("no Qdrant hosts configured")
	}
	srch := searcher.NewQdrant(cfg.VectorDB.Hosts)

	// Reranker
	var rnk reranker.Reranker
	if cfg.Reranker.Enabled && cfg.Reranker.Endpoint != "" {
		rnk = reranker.NewBGE(cfg.Reranker.Endpoint)
	} else {
		rnk = reranker.NewMock()
	}

	// Vault
	var vlt vault.Client
	if cfg.Vault.Enabled && cfg.Vault.Endpoint != "" {
		vlt = vault.NewHTTP(cfg.Vault.Endpoint, c, cfg.Vault.PermissionTTL)
	} else {
		vlt = vault.NewMock()
	}

	return orchestrator.New(cfg, emb, srch, rnk, vlt), nil
}

// buildStandaloneOrchestrator builds an orchestrator with all-mock backends.
func buildStandaloneOrchestrator() *orchestrator.Orchestrator {
	cfg := config.Defaults()
	emb := embedder.NewMock(1024)
	srch := searcher.NewQdrant(cfg.VectorDB.Hosts)
	rnk := reranker.NewMock()
	vlt := vault.NewMock()
	return orchestrator.New(cfg, emb, srch, rnk, vlt)
}
