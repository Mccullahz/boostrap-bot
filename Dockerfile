FROM golang:1.25.7-alpine AS builder

WORKDIR /app

RUN apk add --no-cache git ca-certificates

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -o boostrap-bot ./cmd/bot

FROM gcr.io/distroless/base-debian12

WORKDIR /app

COPY --from=builder /app/boostrap-bot /app/boostrap-bot
COPY config /app/config

USER nonroot:nonroot

ENTRYPOINT ["/app/boostrap-bot"]

