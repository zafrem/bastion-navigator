package grpc

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/zafrem/bastion-navigator/internal/models"
	"github.com/zafrem/bastion-navigator/internal/orchestrator"
)

// Server is the gRPC server for NavigatorService.
type Server struct {
	UnimplementedNavigatorServiceServer
	orch       *orchestrator.Orchestrator
	grpcServer *grpc.Server
	tls        bool
	port       int
}

// New creates a gRPC server with sensible defaults: keepalive, max message
// size of 16 MB, and a recovery interceptor that maps panics to INTERNAL errors.
func New(orch *orchestrator.Orchestrator, port int) *Server {
	s := &Server{orch: orch, port: port}
	s.grpcServer = s.newServer()
	return s
}

// newServer builds the gRPC server (and registers the service) with any extra options, such as TLS.
func (s *Server) newServer(extra ...grpc.ServerOption) *grpc.Server {
	opts := append([]grpc.ServerOption{
		grpc.MaxRecvMsgSize(16 * 1024 * 1024),
		grpc.MaxSendMsgSize(16 * 1024 * 1024),
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 5 * time.Minute,
			Time:              1 * time.Minute,
			Timeout:           20 * time.Second,
		}),
		grpc.ChainUnaryInterceptor(
			loggingInterceptor,
			recoveryInterceptor,
		),
	}, extra...)
	g := grpc.NewServer(opts...)
	RegisterNavigatorServiceServer(g, s)
	reflection.Register(g) // enables grpcurl/grpc-cli introspection
	return g
}

// EnableTLS serves gRPC over TLS using the given PEM files. With all three empty it does nothing and
// the server stays plaintext. clientCAFile (optional) requires client certificates (mutual TLS).
// Call it before Start/Serve; it returns an error for incomplete or unreadable configuration.
func (s *Server) EnableTLS(certFile, keyFile, clientCAFile string) error {
	creds, err := serverTLSCredentials(certFile, keyFile, clientCAFile)
	if err != nil {
		return err
	}
	if creds == nil {
		return nil
	}
	s.grpcServer = s.newServer(grpc.Creds(creds))
	s.tls = true
	return nil
}

// TLSEnabled reports whether EnableTLS turned TLS on.
func (s *Server) TLSEnabled() bool { return s.tls }

// Serve accepts connections on lis until the server stops (used by tests with their own listener).
func (s *Server) Serve(lis net.Listener) error { return s.grpcServer.Serve(lis) }

// Start listens on the configured port and blocks until ctx is cancelled.
func (s *Server) Start(ctx context.Context) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("grpc listen :%d: %w", s.port, err)
	}
	log.Printf("[grpc] listening on :%d", s.port)

	errCh := make(chan error, 1)
	go func() {
		if err := s.grpcServer.Serve(lis); err != nil {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		s.grpcServer.GracefulStop()
		return nil
	}
}

// ─── NavigatorServiceServer implementation ────────────────────────────────────

func (s *Server) Search(ctx context.Context, req *models.SearchRequest) (*models.SearchResponse, error) {
	resp, err := s.orch.Search(ctx, *req)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "search: %v", err)
	}
	return &resp, nil
}

func (s *Server) HybridSearch(ctx context.Context, req *models.SearchRequest) (*models.SearchResponse, error) {
	if req.Options == nil {
		req.Options = &models.SearchOptions{}
	}
	req.Options.UseHybrid = true
	return s.Search(ctx, req)
}

func (s *Server) BatchSearch(ctx context.Context, req *models.BatchSearchRequest) (*models.BatchSearchResponse, error) {
	responses := make([]models.SearchResponse, 0, len(req.Queries))
	for _, q := range req.Queries {
		resp, err := s.orch.Search(ctx, q)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "batch search query %s: %v", q.RequestID, err)
		}
		responses = append(responses, resp)
	}
	return &models.BatchSearchResponse{
		RequestID: req.RequestID,
		Results:   responses,
	}, nil
}

func (s *Server) Embed(ctx context.Context, req *models.EmbedRequest) (*models.EmbedResponse, error) {
	vec, err := s.orch.Embed(ctx, req.Text)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "embed: %v", err)
	}
	return &models.EmbedResponse{
		RequestID: req.RequestID,
		Embedding: vec,
		DimCount:  len(vec),
	}, nil
}

func (s *Server) BatchEmbed(ctx context.Context, req *models.BatchEmbedRequest) (*models.BatchEmbedResponse, error) {
	vecs, err := s.orch.EmbedBatch(ctx, req.Texts)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "batch embed: %v", err)
	}
	return &models.BatchEmbedResponse{
		RequestID:  req.RequestID,
		Embeddings: vecs,
	}, nil
}

func (s *Server) Rerank(ctx context.Context, req *models.RerankRequest) (*models.RerankResponse, error) {
	results, err := s.orch.Rerank(ctx, req.Query, req.Candidates, req.TopK)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "rerank: %v", err)
	}
	return &models.RerankResponse{
		RequestID: req.RequestID,
		Results:   results,
	}, nil
}

func (s *Server) GetCollections(ctx context.Context, _ *models.CollectionsRequest) (*models.CollectionsResponse, error) {
	cols, err := s.orch.Collections(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "collections: %v", err)
	}
	return &models.CollectionsResponse{Collections: cols}, nil
}

func (s *Server) Health(_ context.Context, _ *models.HealthRequest) (*models.HealthStatus, error) {
	return &models.HealthStatus{
		Status:  "ok",
		Version: "1.0.0",
		Checks:  map[string]string{"grpc": "up"},
	}, nil
}

// ─── Interceptors ─────────────────────────────────────────────────────────────

func loggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	code := codes.OK
	if err != nil {
		code = status.Code(err)
	}
	log.Printf("[grpc] %s %s (%v)", info.FullMethod, code, time.Since(start))
	return resp, err
}

func recoveryInterceptor(ctx context.Context, req interface{}, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[grpc] panic recovered: %v", r)
			err = status.Errorf(codes.Internal, "internal server error")
		}
	}()
	return handler(ctx, req)
}
