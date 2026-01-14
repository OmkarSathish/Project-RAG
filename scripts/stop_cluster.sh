#!/bin/bash
# Shutdown script for RAG Distributed System

echo "=============================================="
echo "RAG DISTRIBUTED SYSTEM SHUTDOWN"
echo "=============================================="

# Stop all containers
echo "Stopping all containers..."
docker-compose down

echo ""
echo "All services stopped."
echo ""
echo "To also remove data volumes, run:"
echo "  docker-compose down -v"
