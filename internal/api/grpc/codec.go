package grpc

import (
	"encoding/json"

	"google.golang.org/grpc/encoding"
)

func init() { encoding.RegisterCodec(jsonCodec{}) }

type jsonCodec struct{}

func (jsonCodec) Name() string                              { return "proto" }
func (jsonCodec) Marshal(v interface{}) ([]byte, error)     { return json.Marshal(v) }
func (jsonCodec) Unmarshal(data []byte, v interface{}) error { return json.Unmarshal(data, v) }
