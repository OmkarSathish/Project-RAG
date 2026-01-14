#!/bin/bash
# RabbitMQ Cluster Join Script

echo "Waiting for primary RabbitMQ node..."
sleep 15

echo "Stopping app..."
rabbitmqctl stop_app

echo "Resetting node..."
rabbitmqctl reset

echo "Joining cluster..."
rabbitmqctl join_cluster rabbit@rabbitmq-node-1

echo "Starting app..."
rabbitmqctl start_app

echo "Cluster status:"
rabbitmqctl cluster_status

echo "RabbitMQ node joined cluster successfully!"
