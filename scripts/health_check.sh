#!/bin/bash
# Health check script for RAG Distributed System

echo "=============================================="
echo "RAG CLUSTER HEALTH CHECK"
echo "=============================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check_service() {
    local name=$1
    local url=$2
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "$name: ${GREEN}HEALTHY${NC}"
        return 0
    else
        echo -e "$name: ${RED}UNHEALTHY${NC}"
        return 1
    fi
}

echo ""
echo "Qdrant Cluster:"
check_service "  Node 1" "http://localhost:6333/readyz"
check_service "  Node 2" "http://localhost:6336/readyz"
check_service "  Node 3" "http://localhost:6338/readyz"

echo ""
echo "Redis Cluster:"
for port in 6379 6380 6381 6382 6383 6384; do
    result=$(redis-cli -p $port ping 2>/dev/null)
    if [ "$result" = "PONG" ]; then
        echo -e "  Node (port $port): ${GREEN}HEALTHY${NC}"
    else
        echo -e "  Node (port $port): ${RED}UNHEALTHY${NC}"
    fi
done

echo ""
echo "MongoDB Replica Set:"
mongo_status=$(docker exec mongo-primary mongosh --quiet --eval "rs.status().members.map(m => ({name: m.name, state: m.stateStr}))" 2>/dev/null)
if [ -n "$mongo_status" ]; then
    echo "$mongo_status" | python3 -m json.tool 2>/dev/null || echo "  Status: Connected"
else
    echo -e "  ${RED}Cannot connect to MongoDB${NC}"
fi

echo ""
echo "RabbitMQ Cluster:"
check_service "  Node 1" "http://localhost:15672"
check_service "  Node 2" "http://localhost:15673"
check_service "  Node 3" "http://localhost:15674"

echo ""
echo "Monitoring Stack:"
check_service "  Prometheus" "http://localhost:9090/-/healthy"
check_service "  Grafana" "http://localhost:3000/api/health"

echo ""
echo "=============================================="
