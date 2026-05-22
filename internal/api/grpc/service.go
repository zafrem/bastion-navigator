// Package grpc implements the NavigatorService gRPC server.
//
// Normally protoc / buf would generate this file from proto/navigator/v1/navigator.proto.
// We write it by hand so the project compiles and runs without the protoc toolchain.
// The service wire format uses JSON (see codec.go) instead of binary protobuf.
package grpc

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/bastion/navigator/internal/models"
)

// NavigatorServiceServer is the server-side interface matching the proto definition.
type NavigatorServiceServer interface {
	Search(context.Context, *models.SearchRequest) (*models.SearchResponse, error)
	HybridSearch(context.Context, *models.SearchRequest) (*models.SearchResponse, error)
	BatchSearch(context.Context, *models.BatchSearchRequest) (*models.BatchSearchResponse, error)
	Embed(context.Context, *models.EmbedRequest) (*models.EmbedResponse, error)
	BatchEmbed(context.Context, *models.BatchEmbedRequest) (*models.BatchEmbedResponse, error)
	Rerank(context.Context, *models.RerankRequest) (*models.RerankResponse, error)
	GetCollections(context.Context, *models.CollectionsRequest) (*models.CollectionsResponse, error)
	Health(context.Context, *models.HealthRequest) (*models.HealthStatus, error)
}

// UnimplementedNavigatorServiceServer provides default "Unimplemented" responses
// for every RPC so embedding it in a concrete server satisfies the interface
// even when some methods are intentionally not overridden.
type UnimplementedNavigatorServiceServer struct{}

func (UnimplementedNavigatorServiceServer) Search(_ context.Context, _ *models.SearchRequest) (*models.SearchResponse, error) {
	return nil, status.Error(codes.Unimplemented, "Search not implemented")
}
func (UnimplementedNavigatorServiceServer) HybridSearch(_ context.Context, _ *models.SearchRequest) (*models.SearchResponse, error) {
	return nil, status.Error(codes.Unimplemented, "HybridSearch not implemented")
}
func (UnimplementedNavigatorServiceServer) BatchSearch(_ context.Context, _ *models.BatchSearchRequest) (*models.BatchSearchResponse, error) {
	return nil, status.Error(codes.Unimplemented, "BatchSearch not implemented")
}
func (UnimplementedNavigatorServiceServer) Embed(_ context.Context, _ *models.EmbedRequest) (*models.EmbedResponse, error) {
	return nil, status.Error(codes.Unimplemented, "Embed not implemented")
}
func (UnimplementedNavigatorServiceServer) BatchEmbed(_ context.Context, _ *models.BatchEmbedRequest) (*models.BatchEmbedResponse, error) {
	return nil, status.Error(codes.Unimplemented, "BatchEmbed not implemented")
}
func (UnimplementedNavigatorServiceServer) Rerank(_ context.Context, _ *models.RerankRequest) (*models.RerankResponse, error) {
	return nil, status.Error(codes.Unimplemented, "Rerank not implemented")
}
func (UnimplementedNavigatorServiceServer) GetCollections(_ context.Context, _ *models.CollectionsRequest) (*models.CollectionsResponse, error) {
	return nil, status.Error(codes.Unimplemented, "GetCollections not implemented")
}
func (UnimplementedNavigatorServiceServer) Health(_ context.Context, _ *models.HealthRequest) (*models.HealthStatus, error) {
	return nil, status.Error(codes.Unimplemented, "Health not implemented")
}

// RegisterNavigatorServiceServer attaches srv to the gRPC server s.
func RegisterNavigatorServiceServer(s *grpc.Server, srv NavigatorServiceServer) {
	s.RegisterService(&navigatorServiceDesc, srv)
}

// navigatorServiceDesc is the hand-written equivalent of what protoc-gen-go-grpc generates.
var navigatorServiceDesc = grpc.ServiceDesc{
	ServiceName: "bastion.navigator.v1.NavigatorService",
	HandlerType: (*NavigatorServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{MethodName: "Search", Handler: _Search_Handler},
		{MethodName: "HybridSearch", Handler: _HybridSearch_Handler},
		{MethodName: "BatchSearch", Handler: _BatchSearch_Handler},
		{MethodName: "Embed", Handler: _Embed_Handler},
		{MethodName: "BatchEmbed", Handler: _BatchEmbed_Handler},
		{MethodName: "Rerank", Handler: _Rerank_Handler},
		{MethodName: "GetCollections", Handler: _GetCollections_Handler},
		{MethodName: "Health", Handler: _Health_Handler},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "navigator/v1/navigator.proto",
}

// ─── Unary handler functions ──────────────────────────────────────────────────

func _Search_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.SearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).Search(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/Search"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).Search(ctx, req.(*models.SearchRequest))
	})
}

func _HybridSearch_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.SearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).HybridSearch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/HybridSearch"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).HybridSearch(ctx, req.(*models.SearchRequest))
	})
}

func _BatchSearch_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.BatchSearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).BatchSearch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/BatchSearch"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).BatchSearch(ctx, req.(*models.BatchSearchRequest))
	})
}

func _Embed_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.EmbedRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).Embed(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/Embed"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).Embed(ctx, req.(*models.EmbedRequest))
	})
}

func _BatchEmbed_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.BatchEmbedRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).BatchEmbed(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/BatchEmbed"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).BatchEmbed(ctx, req.(*models.BatchEmbedRequest))
	})
}

func _Rerank_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.RerankRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).Rerank(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/Rerank"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).Rerank(ctx, req.(*models.RerankRequest))
	})
}

func _GetCollections_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.CollectionsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).GetCollections(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/GetCollections"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).GetCollections(ctx, req.(*models.CollectionsRequest))
	})
}

func _Health_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(models.HealthRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(NavigatorServiceServer).Health(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/bastion.navigator.v1.NavigatorService/Health"}
	return interceptor(ctx, in, info, func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(NavigatorServiceServer).Health(ctx, req.(*models.HealthRequest))
	})
}
