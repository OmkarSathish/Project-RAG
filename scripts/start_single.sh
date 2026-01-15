#!/bin/bash
# ============================================
# Start RAG System in Single-Node Mode
# ============================================
# This script starts the simplified single-node setup
# suitable for local development and testing.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "RAG System - Single Node Mode"
echo "============================================"

cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Start single-node infrastructure
echo ""
echo "[1/3] Starting infrastructure services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "[2/3] Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
echo "Service Status:"
docker-compose ps

# Start the application
echo ""
echo "[3/3] Starting RAG application..."
echo ""
echo "Environment: RAG_MODE=single"
echo "API Server: http://localhost:8000"
echo "Qdrant:     http://localhost:6333/dashboard"
echo "RabbitMQ:   http://localhost:15672 (raguser/ragpass)"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000 (admin/admin)"
echo ""

# Export environment and start server
export RAG_MODE=single
source .venv/bin/activate 2>/dev/null || python -m venv .venv && source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
