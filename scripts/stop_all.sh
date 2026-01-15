#!/bin/bash
# ============================================
# Stop All RAG System Services
# ============================================
# Stops both single-node and distributed services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Stopping RAG System Services"
echo "============================================"

cd "$PROJECT_ROOT"

echo ""
echo "Stopping single-node services..."
docker-compose down 2>/dev/null || true

echo ""
echo "Stopping distributed services..."
docker-compose -f docker-compose.distributed.yml down 2>/dev/null || true

echo ""
echo "All services stopped."
echo ""
echo "To remove volumes as well, run:"
echo "  docker-compose down -v"
echo "  docker-compose -f docker-compose.distributed.yml down -v"
