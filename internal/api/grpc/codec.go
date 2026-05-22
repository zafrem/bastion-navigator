package grpc

// jsonCodec replaces the default protobuf codec so the gRPC server can use
// plain Go structs with JSON marshalling instead of generated protobuf types.
// Clients must set Content-Type: application/grpc+json (or send without a
// sub-type, which also routes here after registration).
//
// Registration happens in init() so any import of this package activates it.

import (
	"encoding/json"
	"fmt"
	"math"

	"google.golang.org/grpc/encoding"
)

const codecName = "proto" // Overrides the default protobuf codec.

func init() {
	encoding.RegisterCodec(jsonCodec{})
}

type jsonCodec struct{}

func (jsonCodec) Name() string { return codecName }

func (jsonCodec) Marshal(v interface{}) ([]byte, error) {
	// grpc passes a *codec.bufferPool wrapping the actual message; unwrap it.
	if vv, ok := v.(interface{ GetValue() interface{} }); ok {
		v = vv.GetValue()
	}
	out, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("grpc json codec marshal: %w", err)
	}
	// grpc length-prefix frame can handle at most math.MaxUint32 bytes.
	if len(out) > math.MaxUint32 {
		return nil, fmt.Errorf("grpc json codec: message too large (%d bytes)", len(out))
	}
	return out, nil
}

func (jsonCodec) Unmarshal(data []byte, v interface{}) error {
	if vv, ok := v.(interface{ SetValue(interface{}) }); ok {
		var dst interface{}
		if err := json.Unmarshal(data, &dst); err != nil {
			return err
		}
		vv.SetValue(dst)
		return nil
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("grpc json codec unmarshal into %T: %w", v, err)
	}
	return nil
}
