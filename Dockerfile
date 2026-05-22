FROM golang:1.21-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /bin/navigator-cli ./cmd/navigator-cli

FROM alpine:3.19
RUN apk add --no-cache ca-certificates curl
COPY --from=builder /bin/navigator-cli /usr/local/bin/navigator-cli
EXPOSE 8080 9090 9091
ENTRYPOINT ["navigator-cli"]
CMD ["server", "--config", "/etc/navigator/config.yaml"]
