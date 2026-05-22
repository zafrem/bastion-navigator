package config

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration object.
type Config struct {
	Version        string              `yaml:"version"`
	Server         ServerConfig        `yaml:"server"`
	VectorDB       VectorDBConfig      `yaml:"vector_db"`
	Embedder       EmbedderConfig      `yaml:"embedder"`
	Reranker       RerankerConfig      `yaml:"reranker"`
	SearchDefaults SearchDefaultsConfig `yaml:"search_defaults"`
	Vault          VaultConfig         `yaml:"vault"`
	Tracker        TrackerConfig       `yaml:"tracker"`
	Cache          CacheConfig         `yaml:"cache"`
	Logging        LoggingConfig       `yaml:"logging"`
	Metrics        MetricsConfig       `yaml:"metrics"`
}

type ServerConfig struct {
	RESTPort int `yaml:"rest_port"`
	GRPCPort int `yaml:"grpc_port"`
	Workers  int `yaml:"workers"`
}

type VectorDBConfig struct {
	Type        string                     `yaml:"type"`
	Hosts       []string                   `yaml:"hosts"`
	Collections map[string]CollectionConfig `yaml:"collections"`
}

type CollectionConfig struct {
	VectorSize int        `yaml:"vector_size"`
	Distance   string     `yaml:"distance"`
	HNSW       HNSWConfig `yaml:"hnsw"`
}

type HNSWConfig struct {
	M           int `yaml:"m"`
	EfConstruct int `yaml:"ef_construct"`
	EfSearch    int `yaml:"ef_search"`
}

type EmbedderConfig struct {
	Type      string       `yaml:"type"`
	Endpoint  string       `yaml:"endpoint"`
	ModelPath string       `yaml:"model_path"`
	BatchSize int          `yaml:"batch_size"`
	MaxLength int          `yaml:"max_length"`
	Cache     EmbedCacheConfig `yaml:"cache"`
}

type EmbedCacheConfig struct {
	Enabled bool          `yaml:"enabled"`
	TTL     time.Duration `yaml:"ttl"`
	MaxSize int           `yaml:"max_size"`
}

type RerankerConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Type      string `yaml:"type"`
	Endpoint  string `yaml:"endpoint"`
	ModelPath string `yaml:"model_path"`
	BatchSize int    `yaml:"batch_size"`
	MaxLength int    `yaml:"max_length"`
}

type SearchDefaultsConfig struct {
	TopK                int     `yaml:"top_k"`
	OverFetchMultiplier int     `yaml:"over_fetch_multiplier"`
	UseHybrid           bool    `yaml:"use_hybrid"`
	UseReranking        bool    `yaml:"use_reranking"`
	VectorWeight        float64 `yaml:"vector_weight"`
	BM25Weight          float64 `yaml:"bm25_weight"`
	MinScore            float64 `yaml:"min_score"`
	TimeoutMs           int     `yaml:"timeout_ms"`
}

type VaultConfig struct {
	Enabled           bool          `yaml:"enabled"`
	Endpoint          string        `yaml:"endpoint"`
	CachePermissions  bool          `yaml:"cache_permissions"`
	PermissionTTL     time.Duration `yaml:"permission_ttl"`
}

type TrackerConfig struct {
	Enabled  bool   `yaml:"enabled"`
	Endpoint string `yaml:"endpoint"`
	Async    bool   `yaml:"async"`
}

type CacheConfig struct {
	Type               string        `yaml:"type"`
	URL                string        `yaml:"url"`
	QueryCacheTTL      time.Duration `yaml:"query_cache_ttl"`
	PermissionCacheTTL time.Duration `yaml:"permission_cache_ttl"`
}

type LoggingConfig struct {
	Level  string `yaml:"level"`
	Format string `yaml:"format"`
}

type MetricsConfig struct {
	Enabled bool `yaml:"enabled"`
	Port    int  `yaml:"port"`
}

// Defaults returns a Config with sensible defaults for standalone mode.
func Defaults() *Config {
	return &Config{
		Version: "1.0",
		Server: ServerConfig{
			RESTPort: 8080,
			GRPCPort: 9090,
			Workers:  4,
		},
		VectorDB: VectorDBConfig{
			Type:  "qdrant",
			Hosts: []string{"localhost:6333"},
			Collections: map[string]CollectionConfig{
				"customer_docs": {
					VectorSize: 1024,
					Distance:   "cosine",
					HNSW:       HNSWConfig{M: 16, EfConstruct: 200, EfSearch: 128},
				},
				"manufacturing_docs": {
					VectorSize: 1024,
					Distance:   "cosine",
					HNSW:       HNSWConfig{M: 16, EfConstruct: 200, EfSearch: 128},
				},
				"hr_docs": {
					VectorSize: 1024,
					Distance:   "cosine",
					HNSW:       HNSWConfig{M: 16, EfConstruct: 200, EfSearch: 128},
				},
			},
		},
		Embedder: EmbedderConfig{
			Type:      "bge_m3",
			Endpoint:  "http://localhost:8000",
			BatchSize: 16,
			MaxLength: 512,
			Cache: EmbedCacheConfig{
				Enabled: true,
				TTL:     time.Hour,
				MaxSize: 10000,
			},
		},
		Reranker: RerankerConfig{
			Enabled:   true,
			Type:      "bge_reranker",
			Endpoint:  "http://localhost:8001",
			BatchSize: 16,
			MaxLength: 512,
		},
		SearchDefaults: SearchDefaultsConfig{
			TopK:                10,
			OverFetchMultiplier: 5,
			UseHybrid:           true,
			UseReranking:        true,
			VectorWeight:        0.7,
			BM25Weight:          0.3,
			MinScore:            0.5,
			TimeoutMs:           500,
		},
		Vault: VaultConfig{
			Enabled:          false,
			Endpoint:         "http://localhost:8081",
			CachePermissions: true,
			PermissionTTL:    5 * time.Minute,
		},
		Tracker: TrackerConfig{
			Enabled:  false,
			Endpoint: "http://localhost:8082",
			Async:    true,
		},
		Cache: CacheConfig{
			Type:               "redis",
			URL:                "redis://localhost:6379",
			QueryCacheTTL:      time.Hour,
			PermissionCacheTTL: 5 * time.Minute,
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "json",
		},
		Metrics: MetricsConfig{
			Enabled: true,
			Port:    9091,
		},
	}
}

// Load reads config from the given YAML file, falling back to defaults for missing fields.
func Load(path string) (*Config, error) {
	cfg := Defaults()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}

	return cfg, nil
}
