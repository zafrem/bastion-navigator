package grpc

import "encoding/json"

// jsonCodec carries the service's JSON-encoded messages. It is installed on this server only, via
// grpc.ForceServerCodec: it is deliberately not registered under the name "proto", which would replace
// the process-wide protobuf codec for every other gRPC service and client in the binary.
type jsonCodec struct{}

func (jsonCodec) Name() string                               { return "json" }
func (jsonCodec) Marshal(v interface{}) ([]byte, error)      { return json.Marshal(v) }
func (jsonCodec) Unmarshal(data []byte, v interface{}) error { return json.Unmarshal(data, v) }
