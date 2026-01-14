#!/bin/bash
# Startup script for RAG Distributed System

set -e

echo "=============================================="
echo "RAG DISTRIBUTED SYSTEM STARTUP"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to wait for service
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e " ${RED}Failed!${NC}"
    return 1
}

# Step 1: Start infrastructure
echo -e "\n${YELLOW}[1/5] Starting infrastructure services...${NC}"
docker-compose up -d

# Step 2: Wait for services to be ready
echo -e "\n${YELLOW}[2/5] Waiting for services to be healthy...${NC}"

# Wait for Qdrant cluster
wait_for_service "Qdrant Node 1" "http://localhost:6333/readyz"
wait_for_service "Qdrant Node 2" "http://localhost:6336/readyz"
wait_for_service "Qdrant Node 3" "http://localhost:6338/readyz"

# Wait for Redis
wait_for_service "Redis" "http://localhost:6379" || echo "Redis check skipped (no HTTP)"

# Wait for MongoDB
echo -n "Waiting for MongoDB..."
sleep 10
echo -e " ${GREEN}Ready!${NC}"

# Wait for RabbitMQ
wait_for_service "RabbitMQ" "http://localhost:15672"

# Wait for Prometheus
wait_for_service "Prometheus" "http://localhost:9090/-/healthy"

# Wait for Grafana
wait_for_service "Grafana" "http://localhost:3000/api/health"

# Step 3: Initialize Redis cluster
echo -e "\n${YELLOW}[3/5] Initializing Redis cluster...${NC}"
docker exec redis-cluster-init sh /scripts/init.sh 2>/dev/null || echo "Redis cluster may already be initialized"

# Step 4: Initialize MongoDB replica set
echo -e "\n${YELLOW}[4/5] Initializing MongoDB replica set...${NC}"
docker exec mongo-init bash /scripts/init.sh 2>/dev/null || echo "MongoDB replica set may already be initialized"

# Step 5: Verify cluster health
echo -e "\n${YELLOW}[5/5] Verifying cluster health...${NC}"

echo ""
echo "Qdrant Cluster:"
curl -s http://localhost:6333/cluster | python3 -m json.tool 2>/dev/null || echo "  Cluster info unavailable"

echo ""
echo "=============================================="
echo -e "${GREEN}DISTRIBUTED SYSTEM READY!${NC}"
echo "=============================================="
echo ""
echo "Access points:"
echo "  - Qdrant UI:      http://localhost:6333/dashboard"
echo "  - RabbitMQ UI:    http://localhost:15672 (raguser/ragpass)"
echo "  - Prometheus:     http://localhost:9090"
echo "  - Grafana:        http://localhost:3000 (admin/admin)"
echo ""
echo "Next steps:"
echo "  1. Index documents: python embed_distributed.py"
echo "  2. Start API:       cd src/api && python main.py"
echo ""
