package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache is the storage interface used throughout Navigator.
type Cache interface {
	GetEmbedding(ctx context.Context, text string) ([]float32, bool)
	SetEmbedding(ctx context.Context, text string, vec []float32, ttl time.Duration) error
	GetPermissions(ctx context.Context, userID string) ([]string, bool)
	SetPermissions(ctx context.Context, userID string, cats []string, ttl time.Duration) error
	Close() error
}

// keyFor generates a deterministic cache key for any string.
func keyFor(prefix, text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("nav:%s:%x", prefix, h[:8])
}

// RedisCache is the production-grade cache backed by Redis.
type RedisCache struct {
	client *redis.Client
}

func NewRedis(url string) (*RedisCache, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("parse redis url: %w", err)
	}
	c := redis.NewClient(opts)
	if err := c.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}
	return &RedisCache{client: c}, nil
}

func (r *RedisCache) GetEmbedding(ctx context.Context, text string) ([]float32, bool) {
	key := keyFor("emb", text)
	data, err := r.client.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}
	var vec []float32
	if err := json.Unmarshal(data, &vec); err != nil {
		return nil, false
	}
	return vec, true
}

func (r *RedisCache) SetEmbedding(ctx context.Context, text string, vec []float32, ttl time.Duration) error {
	data, err := json.Marshal(vec)
	if err != nil {
		return err
	}
	return r.client.Set(ctx, keyFor("emb", text), data, ttl).Err()
}

func (r *RedisCache) GetPermissions(ctx context.Context, userID string) ([]string, bool) {
	key := keyFor("perm", userID)
	data, err := r.client.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}
	var cats []string
	if err := json.Unmarshal(data, &cats); err != nil {
		return nil, false
	}
	return cats, true
}

func (r *RedisCache) SetPermissions(ctx context.Context, userID string, cats []string, ttl time.Duration) error {
	data, err := json.Marshal(cats)
	if err != nil {
		return err
	}
	return r.client.Set(ctx, keyFor("perm", userID), data, ttl).Err()
}

func (r *RedisCache) Close() error {
	return r.client.Close()
}

// InMemoryCache is a lightweight fallback used when Redis is unavailable.
type InMemoryCache struct {
	mu      sync.RWMutex
	entries map[string]inmemEntry
}

type inmemEntry struct {
	data    []byte
	expires time.Time
}

func NewInMemory() *InMemoryCache {
	c := &InMemoryCache{entries: make(map[string]inmemEntry)}
	go c.evictLoop()
	return c
}

func (m *InMemoryCache) get(key string) ([]byte, bool) {
	m.mu.RLock()
	e, ok := m.entries[key]
	m.mu.RUnlock()
	if !ok || time.Now().After(e.expires) {
		return nil, false
	}
	return e.data, true
}

func (m *InMemoryCache) set(key string, data []byte, ttl time.Duration) {
	m.mu.Lock()
	m.entries[key] = inmemEntry{data: data, expires: time.Now().Add(ttl)}
	m.mu.Unlock()
}

func (m *InMemoryCache) GetEmbedding(ctx context.Context, text string) ([]float32, bool) {
	data, ok := m.get(keyFor("emb", text))
	if !ok {
		return nil, false
	}
	var vec []float32
	if err := json.Unmarshal(data, &vec); err != nil {
		return nil, false
	}
	return vec, true
}

func (m *InMemoryCache) SetEmbedding(_ context.Context, text string, vec []float32, ttl time.Duration) error {
	data, err := json.Marshal(vec)
	if err != nil {
		return err
	}
	m.set(keyFor("emb", text), data, ttl)
	return nil
}

func (m *InMemoryCache) GetPermissions(_ context.Context, userID string) ([]string, bool) {
	data, ok := m.get(keyFor("perm", userID))
	if !ok {
		return nil, false
	}
	var cats []string
	if err := json.Unmarshal(data, &cats); err != nil {
		return nil, false
	}
	return cats, true
}

func (m *InMemoryCache) SetPermissions(_ context.Context, userID string, cats []string, ttl time.Duration) error {
	data, err := json.Marshal(cats)
	if err != nil {
		return err
	}
	m.set(keyFor("perm", userID), data, ttl)
	return nil
}

func (m *InMemoryCache) Close() error { return nil }

func (m *InMemoryCache) evictLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	for range ticker.C {
		now := time.Now()
		m.mu.Lock()
		for k, e := range m.entries {
			if now.After(e.expires) {
				delete(m.entries, k)
			}
		}
		m.mu.Unlock()
	}
}
