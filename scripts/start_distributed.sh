#!/bin/bash
# ============================================
# Start RAG System in Distributed Mode
# ============================================
# This script starts the full distributed cluster setup
# with multi-node Qdrant, MongoDB replica set, Redis cluster, etc.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "RAG System - Distributed Mode"
echo "============================================"

cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing single-node setup
echo ""
echo "[1/5] Stopping any existing single-node services..."
docker-compose down 2>/dev/null || true

# Start distributed infrastructure
echo ""
echo "[2/5] Starting distributed cluster..."
docker-compose -f docker-compose.distributed.yml up -d

# Wait for services to initialize
echo ""
echo "[3/5] Waiting for cluster initialization..."
echo "  (This may take 30-60 seconds for all nodes to join...)"
sleep 10

# Check cluster status
echo ""
echo "[4/5] Checking cluster status..."
echo ""
echo "Qdrant Cluster:"
docker-compose -f docker-compose.distributed.yml ps | grep qdrant || true

echo ""
echo "MongoDB Replica Set:"
docker-compose -f docker-compose.distributed.yml ps | grep mongo || true

echo ""
echo "Redis Cluster:"
docker-compose -f docker-compose.distributed.yml ps | grep redis || true

echo ""
echo "RabbitMQ Cluster:"
docker-compose -f docker-compose.distributed.yml ps | grep rabbitmq || true

# Start the application
echo ""
echo "[5/5] Starting RAG application in distributed mode..."
echo ""
echo "Environment: RAG_MODE=distributed"
echo ""
echo "Endpoints:"
echo "  API Server:    http://localhost:8000"
echo "  Qdrant Node 1: http://localhost:6333/dashboard"
echo "  Qdrant Node 2: http://localhost:6336/dashboard"
echo "  Qdrant Node 3: http://localhost:6338/dashboard"
echo "  RabbitMQ:      http://localhost:15672 (raguser/ragpass)"
echo "  Prometheus:    http://localhost:9090"
echo "  Grafana:       http://localhost:3000 (admin/admin)"
echo ""

# Export environment and start server
export RAG_MODE=distributed
source .venv/bin/activate 2>/dev/null || python -m venv .venv && source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
