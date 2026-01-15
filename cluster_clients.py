"""
Cluster-Aware Database Clients for Distributed RAG System.

This module provides production-grade client wrappers with:
- Automatic failover and reconnection
- Load balancing across cluster nodes
- Health checking and node discovery
- Connection pooling

Usage:
    from cluster_clients import DistributedClientManager, ClusterConfig
    
    config = ClusterConfig()
    manager = DistributedClientManager(config)
    await manager.connect_all()
"""

import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
import aio_pika


@dataclass
class ClusterConfig:
    """
    Configuration for distributed cluster connections.
    
    For Docker networking, use container names (e.g., 'qdrant-node-1')
    For local development, use localhost with mapped ports
    """
    
    # Qdrant cluster nodes
    qdrant_nodes: List[str] = field(default_factory=list)
    
    # MongoDB replica set URI
    mongodb_uri: str = ""
    
    # Redis cluster nodes
    redis_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    # RabbitMQ cluster nodes
    rabbitmq_nodes: List[str] = field(default_factory=list)
    
    # Use Docker internal networking (container names)
    use_docker_network: bool = False
    
    def __post_init__(self):
        if not self.qdrant_nodes:
            if self.use_docker_network:
                self.qdrant_nodes = [
                    "http://qdrant-node-1:6333",
                    "http://qdrant-node-2:6333",
                    "http://qdrant-node-3:6333",
                ]
            else:
                # Local development - use localhost with mapped ports
                self.qdrant_nodes = [
                    "http://localhost:6333",
                    "http://localhost:6336",
                    "http://localhost:6338",
                ]
        
        if not self.mongodb_uri:
            if self.use_docker_network:
                self.mongodb_uri = "mongodb://mongo-primary:27017,mongo-secondary-1:27017,mongo-secondary-2:27017/?replicaSet=rs0"
            else:
                self.mongodb_uri = "mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=rs0"
        
        if not self.redis_nodes:
            if self.use_docker_network:
                self.redis_nodes = [
                    {"host": "redis-node-1", "port": 6379},
                    {"host": "redis-node-2", "port": 6379},
                    {"host": "redis-node-3", "port": 6379},
                    {"host": "redis-node-4", "port": 6379},
                    {"host": "redis-node-5", "port": 6379},
                    {"host": "redis-node-6", "port": 6379},
                ]
            else:
                self.redis_nodes = [
                    {"host": "localhost", "port": 6379},
                    {"host": "localhost", "port": 6380},
                    {"host": "localhost", "port": 6381},
                    {"host": "localhost", "port": 6382},
                    {"host": "localhost", "port": 6383},
                    {"host": "localhost", "port": 6384},
                ]
        
        if not self.rabbitmq_nodes:
            if self.use_docker_network:
                self.rabbitmq_nodes = [
                    "amqp://raguser:ragpass@rabbitmq-node-1:5672/",
                    "amqp://raguser:ragpass@rabbitmq-node-2:5672/",
                    "amqp://raguser:ragpass@rabbitmq-node-3:5672/",
                ]
            else:
                self.rabbitmq_nodes = [
                    "amqp://raguser:ragpass@localhost:5672/",
                    "amqp://raguser:ragpass@localhost:5673/",
                    "amqp://raguser:ragpass@localhost:5674/",
                ]


class QdrantClusterClient:
    """
    Cluster-aware Qdrant client with load balancing and failover.
    
    Features:
    - Round-robin load balancing across nodes
    - Automatic failover on node failure
    - Health checking before operations
    - Consistent hashing for shard routing
    """
    
    def __init__(self, nodes: List[str]):
        """
        Initialize Qdrant cluster client.
        
        Args:
            nodes: List of Qdrant node URLs
        """
        self.nodes = nodes
        self.clients: Dict[str, AsyncQdrantClient] = {}
        self.healthy_nodes: List[str] = []
        self.current_index = 0
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Initialize connections to all nodes and perform health check"""
        for node_url in self.nodes:
            try:
                client = AsyncQdrantClient(url=node_url, timeout=10)
                # Health check
                await client.get_collections()
                self.clients[node_url] = client
                self.healthy_nodes.append(node_url)
                print(f"[QdrantCluster] Connected to {node_url}")
            except Exception as e:
                print(f"[QdrantCluster] Failed to connect to {node_url}: {e}")
        
        if not self.healthy_nodes:
            raise ConnectionError("No healthy Qdrant nodes available")
        
        print(f"[QdrantCluster] {len(self.healthy_nodes)}/{len(self.nodes)} nodes healthy")
    
    async def get_client(self) -> AsyncQdrantClient:
        """Get a healthy client using round-robin selection"""
        async with self._lock:
            if not self.healthy_nodes:
                await self._refresh_health()
            
            node_url = self.healthy_nodes[self.current_index % len(self.healthy_nodes)]
            self.current_index += 1
            return self.clients[node_url]
    
    async def get_client_for_key(self, key: str) -> AsyncQdrantClient:
        """
        Get client based on consistent hashing for shard affinity.
        
        Args:
            key: Key to hash for consistent routing
        """
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        node_index = hash_value % len(self.healthy_nodes)
        node_url = self.healthy_nodes[node_index]
        return self.clients[node_url]
    
    async def _refresh_health(self):
        """Refresh health status of all nodes"""
        self.healthy_nodes = []
        for node_url, client in self.clients.items():
            try:
                await client.get_collections()
                self.healthy_nodes.append(node_url)
            except Exception:
                print(f"[QdrantCluster] Node {node_url} unhealthy")
        
        if not self.healthy_nodes:
            raise ConnectionError("No healthy Qdrant nodes available")
    
    async def search(self, collection_name: str, query_vector: List[float], limit: int = 10):
        """
        Perform distributed search with automatic failover.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results
        """
        retries = len(self.healthy_nodes)
        last_error = None
        
        for _ in range(retries):
            try:
                client = await self.get_client()
                results = await client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                )
                return results
            except Exception as e:
                last_error = e
                await self._refresh_health()
        
        raise last_error
    
    async def close(self):
        """Close all client connections"""
        for client in self.clients.values():
            await client.close()
        print("[QdrantCluster] All connections closed")


class RedisClusterClient:
    """
    Redis Cluster client with automatic sharding and failover.
    
    Features:
    - Automatic slot routing (16384 hash slots)
    - Read replicas for scaling reads
    - Automatic failover on master failure
    """
    
    def __init__(self, nodes: List[Dict[str, Any]]):
        """
        Initialize Redis cluster client.
        
        Args:
            nodes: List of node configurations with host and port
        """
        self.nodes = nodes
        self.cluster: Optional[RedisCluster] = None
        self._fallback_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis cluster or fallback to single node"""
        try:
            # Try connecting as cluster
            startup_nodes = [
                redis.cluster.ClusterNode(n["host"], n["port"]) 
                for n in self.nodes
            ]
            self.cluster = RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=True,
                read_from_replicas=True,  # Enable read scaling
            )
            await self.cluster.initialize()
            print(f"[RedisCluster] Connected to cluster with {len(self.nodes)} nodes")
        except Exception as e:
            # Fallback to single node (for development)
            print(f"[RedisCluster] Cluster connection failed, using single node: {e}")
            self._fallback_client = await redis.from_url(
                f"redis://{self.nodes[0]['host']}:{self.nodes[0]['port']}"
            )
    
    @property
    def client(self):
        """Get the active client (cluster or fallback)"""
        return self.cluster if self.cluster else self._fallback_client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cluster with automatic routing"""
        return await self.client.get(key)
    
    async def set(self, key: str, value: str, ex: int = None):
        """Set value in cluster with automatic routing"""
        if ex:
            await self.client.setex(key, ex, value)
        else:
            await self.client.set(key, value)
    
    async def setex(self, key: str, seconds: int, value: str):
        """Set value with expiration"""
        await self.client.setex(key, seconds, value)
    
    async def delete(self, key: str):
        """Delete key from cluster"""
        await self.client.delete(key)
    
    async def close(self):
        """Close cluster connection"""
        if self.cluster:
            await self.cluster.close()
        if self._fallback_client:
            await self._fallback_client.close()
        print("[RedisCluster] Connection closed")


class MongoDBReplicaSetClient:
    """
    MongoDB Replica Set client with read preference configuration.
    
    Features:
    - Automatic failover to secondary
    - Read preference (primary, secondary, nearest)
    - Write concern configuration
    - Connection pooling
    """
    
    def __init__(self, uri: str):
        """
        Initialize MongoDB replica set client.
        
        Args:
            uri: MongoDB connection URI with replica set
        """
        self.uri = uri
        self.client: Optional[AsyncIOMotorClient] = None
    
    async def connect(self, 
                      read_preference: str = "primaryPreferred",
                      max_pool_size: int = 100):
        """
        Connect to MongoDB replica set.
        
        Args:
            read_preference: Read preference mode
                - "primary": Always read from primary
                - "primaryPreferred": Prefer primary, fallback to secondary
                - "secondary": Always read from secondary
                - "secondaryPreferred": Prefer secondary, fallback to primary
                - "nearest": Read from nearest node (lowest latency)
            max_pool_size: Maximum connection pool size
        """
        self.client = AsyncIOMotorClient(
            self.uri,
            readPreference=read_preference,
            maxPoolSize=max_pool_size,
            w="majority",  # Write concern: majority for durability
            wtimeout=5000,  # Write timeout in ms
            retryWrites=True,
            retryReads=True,
        )
        
        # Verify connection
        await self.client.admin.command('ping')
        print(f"[MongoDBReplicaSet] Connected with read preference: {read_preference}")
    
    def get_database(self, db_name: str):
        """Get database reference"""
        return self.client[db_name]
    
    def get_collection(self, db_name: str, collection_name: str):
        """Get collection reference"""
        return self.client[db_name][collection_name]
    
    async def get_replica_status(self) -> Dict[str, Any]:
        """Get replica set status for monitoring"""
        return await self.client.admin.command('replSetGetStatus')
    
    async def close(self):
        """Close client connection"""
        if self.client:
            self.client.close()
        print("[MongoDBReplicaSet] Connection closed")


class RabbitMQClusterClient:
    """
    RabbitMQ Cluster client with failover.
    
    Features:
    - Automatic reconnection on failure
    - Load balancing across nodes
    - Publisher confirms for reliability
    """
    
    def __init__(self, nodes: List[str]):
        """
        Initialize RabbitMQ cluster client.
        
        Args:
            nodes: List of AMQP URLs for cluster nodes
        """
        self.nodes = nodes
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self._current_node_index = 0
    
    async def connect(self):
        """Connect to RabbitMQ cluster with failover"""
        last_error = None
        
        for i, node_url in enumerate(self.nodes):
            try:
                self.connection = await aio_pika.connect_robust(
                    node_url,
                    heartbeat=60,
                    connection_attempts=3,
                    retry_delay=1,
                )
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=1)
                self._current_node_index = i
                print(f"[RabbitMQCluster] Connected to node: {node_url}")
                return
            except Exception as e:
                last_error = e
                print(f"[RabbitMQCluster] Failed to connect to {node_url}: {e}")
        
        raise ConnectionError(f"Failed to connect to any RabbitMQ node: {last_error}")
    
    async def declare_queue(self, queue_name: str, durable: bool = True):
        """Declare a queue"""
        return await self.channel.declare_queue(queue_name, durable=durable)
    
    async def publish(self, queue_name: str, message: bytes, priority: int = 5):
        """Publish message to queue with publisher confirms"""
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=message,
                priority=priority,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue_name,
        )
    
    async def close(self):
        """Close connection"""
        if self.connection:
            await self.connection.close()
        print("[RabbitMQCluster] Connection closed")


class DistributedClientManager:
    """
    Unified manager for all distributed database clients.
    
    Provides a single interface to initialize and access all cluster clients.
    """
    
    def __init__(self, config: ClusterConfig = None):
        """
        Initialize distributed client manager.
        
        Args:
            config: Cluster configuration (uses defaults if not provided)
        """
        self.config = config or ClusterConfig()
        self.qdrant: Optional[QdrantClusterClient] = None
        self.redis: Optional[RedisClusterClient] = None
        self.mongodb: Optional[MongoDBReplicaSetClient] = None
        self.rabbitmq: Optional[RabbitMQClusterClient] = None
    
    async def connect_all(self):
        """Initialize all cluster connections"""
        print("[DistributedClientManager] Initializing cluster connections...")
        
        # Connect to all clusters in parallel
        tasks = []
        
        # Qdrant cluster
        self.qdrant = QdrantClusterClient(self.config.qdrant_nodes)
        tasks.append(self.qdrant.connect())
        
        # Redis cluster
        self.redis = RedisClusterClient(self.config.redis_nodes)
        tasks.append(self.redis.connect())
        
        # MongoDB replica set
        self.mongodb = MongoDBReplicaSetClient(self.config.mongodb_uri)
        tasks.append(self.mongodb.connect())
        
        # RabbitMQ cluster
        self.rabbitmq = RabbitMQClusterClient(self.config.rabbitmq_nodes)
        tasks.append(self.rabbitmq.connect())
        
        await asyncio.gather(*tasks)
        print("[DistributedClientManager] All clusters connected!")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all clusters"""
        health = {}
        
        try:
            client = await self.qdrant.get_client()
            await client.get_collections()
            health["qdrant"] = True
        except Exception:
            health["qdrant"] = False
        
        try:
            await self.redis.client.ping()
            health["redis"] = True
        except Exception:
            health["redis"] = False
        
        try:
            await self.mongodb.client.admin.command('ping')
            health["mongodb"] = True
        except Exception:
            health["mongodb"] = False
        
        try:
            if self.rabbitmq.connection and not self.rabbitmq.connection.is_closed:
                health["rabbitmq"] = True
            else:
                health["rabbitmq"] = False
        except Exception:
            health["rabbitmq"] = False
        
        return health
    
    async def close_all(self):
        """Close all cluster connections"""
        await asyncio.gather(
            self.qdrant.close() if self.qdrant else asyncio.sleep(0),
            self.redis.close() if self.redis else asyncio.sleep(0),
            self.mongodb.close() if self.mongodb else asyncio.sleep(0),
            self.rabbitmq.close() if self.rabbitmq else asyncio.sleep(0),
        )
        print("[DistributedClientManager] All connections closed")


# Singleton instance for application use
_distributed_clients: Optional[DistributedClientManager] = None


async def get_distributed_clients() -> DistributedClientManager:
    """Get or create the distributed client manager singleton"""
    global _distributed_clients
    if _distributed_clients is None:
        _distributed_clients = DistributedClientManager()
        await _distributed_clients.connect_all()
    return _distributed_clients
