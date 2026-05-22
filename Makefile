.PHONY: all build test lint generate clean docker run-standalone

BINARY := navigator-cli
MODULE  := github.com/bastion/navigator
CMD     := ./cmd/navigator-cli

all: generate build

build:
	go build -ldflags="-s -w" -o bin/$(BINARY) $(CMD)

test:
	go test ./... -race -timeout 60s

lint:
	golangci-lint run ./...

# Requires buf (https://buf.build/docs/installation) or protoc with go plugins.
generate:
	@which buf > /dev/null 2>&1 && buf generate || \
	 (echo "buf not found, trying protoc..." && \
	  protoc --go_out=. --go_opt=paths=source_relative \
	         --go-grpc_out=. --go-grpc_opt=paths=source_relative \
	         proto/navigator/v1/navigator.proto)

clean:
	rm -rf bin/ gen/

docker:
	docker build -t bastion/navigator:dev .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

run-standalone:
	go run $(CMD) server --standalone --port 8080

run-server:
	go run $(CMD) server --config config/config.yaml

# Quick interactive search without a running server.
interactive:
	go run $(CMD) interactive --standalone

tidy:
	go mod tidy

.DEFAULT_GOAL := build
