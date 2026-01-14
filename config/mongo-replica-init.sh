#!/bin/bash
# MongoDB Replica Set Initialization Script

echo "Waiting for MongoDB nodes to be ready..."
sleep 15

echo "Initiating replica set..."
mongosh --host mongo-primary:27017 <<EOF
rs.initiate({
  _id: "rs0",
  members: [
    { _id: 0, host: "mongo-primary:27017", priority: 2 },
    { _id: 1, host: "mongo-secondary-1:27017", priority: 1 },
    { _id: 2, host: "mongo-secondary-2:27017", priority: 1 }
  ]
})
EOF

echo "Waiting for replica set to stabilize..."
sleep 10

echo "Checking replica set status..."
mongosh --host mongo-primary:27017 --eval "rs.status()"

echo "MongoDB Replica Set initialized successfully!"
