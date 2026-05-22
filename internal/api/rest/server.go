package rest

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/bastion/navigator/internal/orchestrator"
)

// Server is the HTTP REST API server.
type Server struct {
	orchestrator *orchestrator.Orchestrator
	port         int
	httpServer   *http.Server
}

func New(orch *orchestrator.Orchestrator, port int) *Server {
	s := &Server{orchestrator: orch, port: port}
	s.httpServer = &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      s.routes(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	return s
}

func (s *Server) routes() http.Handler {
	r := chi.NewRouter()

	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(30 * time.Second))

	h := &handlers{orch: s.orchestrator}

	// Health
	r.Get("/v1/health", h.Health)
	r.Get("/v1/health/live", h.Live)
	r.Get("/v1/health/ready", h.Ready)

	// Metrics (Prometheus)
	r.Handle("/v1/metrics", promhttp.Handler())

	// Search
	r.Post("/v1/navigator/search", h.Search)
	r.Post("/v1/navigator/search/hybrid", h.HybridSearch)
	r.Post("/v1/navigator/search/batch", h.BatchSearch)

	// Embedding
	r.Post("/v1/navigator/embed", h.Embed)
	r.Post("/v1/navigator/embed/batch", h.BatchEmbed)

	// Reranking
	r.Post("/v1/navigator/rerank", h.Rerank)

	// Collections
	r.Get("/v1/navigator/collections", h.Collections)
	r.Get("/v1/navigator/collections/{name}", h.CollectionInfo)

	return r
}

// Start begins listening. It blocks until ctx is cancelled.
func (s *Server) Start(ctx context.Context) error {
	log.Printf("[rest] listening on :%d", s.port)
	errCh := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return s.httpServer.Shutdown(shutdownCtx)
	}
}
