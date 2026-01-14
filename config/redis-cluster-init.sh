#!/bin/sh
# Redis Cluster Initialization Script

echo "Waiting for Redis nodes to be ready..."
sleep 10

echo "Creating Redis cluster..."
redis-cli --cluster create \
  redis-node-1:6379 \
  redis-node-2:6379 \
  redis-node-3:6379 \
  redis-node-4:6379 \
  redis-node-5:6379 \
  redis-node-6:6379 \
  --cluster-replicas 1 \
  --cluster-yes

echo "Checking cluster info..."
redis-cli -h redis-node-1 -p 6379 cluster info

echo "Redis cluster initialized successfully!"
