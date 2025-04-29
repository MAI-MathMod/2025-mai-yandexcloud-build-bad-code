# syntax=docker/dockerfile:1

# СТАДИЯ 1: сборка статического бинарника
FROM golang:1.22 AS builder

WORKDIR /app

COPY internal/go.mod internal/go.sum ./
RUN go mod download

COPY internal/ ./internal

WORKDIR /app/internal

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64

RUN go build -a -installsuffix cgo -o bot_server_service bot_server_service.go bot_client_service.go


# СТАДИЯ 2: минимальный образ с CA-cert для TLS
FROM alpine:3.18

WORKDIR /app

RUN apk add --no-cache ca-certificates

COPY --from=builder /app/internal/bot_server_service ./bot_server_service

ENV BOT_TOKEN="8177325212:AAHyEHDl54YuuFCf6s3aQqmDttEad0QVikA"

ENV YC_SERVICE_ACCOUNT_ID="YCAJED2FDEw2PcU0hzFK8m4zk"

ENV YC_IAM_TOKEN="YCMrkYtwQ6tLldsUr6mpBVEtsADPe3bY1H5vgfuM"

ENTRYPOINT ["./bot_server_service"]

