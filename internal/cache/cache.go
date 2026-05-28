// Package cache provides a simple key-value cache interface with in-memory
// and Redis-stub backends.
package cache

import (
	"net"
	"strings"
	"sync"
	"time"
)

// Cache is a simple key-value byte store.
type Cache interface {
	Get(key string) ([]byte, bool)
	Set(key string, value []byte)
	Delete(key string)
}

// --- In-memory ---

type inMemory struct {
	mu    sync.RWMutex
	store map[string][]byte
}

// NewInMemory returns a thread-safe in-memory cache.
func NewInMemory() Cache {
	return &inMemory{store: make(map[string][]byte)}
}

func (c *inMemory) Get(key string) ([]byte, bool) {
	c.mu.RLock()
	v, ok := c.store[key]
	c.mu.RUnlock()
	return v, ok
}

func (c *inMemory) Set(key string, value []byte) {
	c.mu.Lock()
	c.store[key] = value
	c.mu.Unlock()
}

func (c *inMemory) Delete(key string) {
	c.mu.Lock()
	delete(c.store, key)
	c.mu.Unlock()
}

// --- Redis (lightweight stub) ---

// redisCache satisfies Cache with no-op reads/writes.
// A real production implementation would use a proper Redis client.
type redisCache struct {
	addr string
	inMemory
}

// NewRedis performs a connectivity check against the Redis URL and returns
// a Cache on success, or an error if the server is unreachable.
func NewRedis(url string) (Cache, error) {
	addr := strings.TrimPrefix(url, "redis://")
	if !strings.Contains(addr, ":") {
		addr += ":6379"
	}
	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		return nil, err
	}
	_ = conn.Close()
	return &redisCache{addr: addr, inMemory: inMemory{store: make(map[string][]byte)}}, nil
}
