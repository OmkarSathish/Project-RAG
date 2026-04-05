"""
Configuration module for RAG system architecture modes.

Supports switching between:
- 'single': Single-node development setup
- 'distributed': Multi-node production cluster setup

Usage:
    from src.core.config import get_config, RAGMode
    
    config = get_config()
    if config.mode == RAGMode.DISTRIBUTED:
        # Use cluster clients
    else:
        # Use single-node clients
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class RAGMode(Enum):
    """RAG system architecture mode"""
    SINGLE = "single"
    DISTRIBUTED = "distributed"


@dataclass
class SingleNodeConfig:
    """Configuration for single-node development setup"""
    qdrant_url: str = "http://localhost:6333"
    mongodb_url: str = "mongodb://localhost:27017"
    redis_url: str = "redis://localhost:6379"
    rabbitmq_url: str = "amqp://raguser:ragpass@localhost:5672/"
    
    # Collection/DB names
    qdrant_collection: str = "nodejs"
    mongodb_database: str = "rag_db"
    
    # Embedding model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072


@dataclass
class DistributedConfig:
    """Configuration for distributed production cluster"""
    
    # Qdrant single node (with sharding support)
    # Note: Multi-node Qdrant clusters have complex initialization requirements
    # For production, use Qdrant Cloud. For local testing, we use single node with sharding.
    qdrant_nodes: List[str] = field(default_factory=lambda: [
        "http://localhost:6333",
    ])
    
    # MongoDB replica set URI
    # When running application outside Docker, use localhost with mapped ports
    mongodb_uri: str = "mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=rs0"
    
    # Redis cluster startup nodes
    redis_nodes: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"host": "localhost", "port": 6379},
        {"host": "localhost", "port": 6380},
        {"host": "localhost", "port": 6381},
        {"host": "localhost", "port": 6382},
        {"host": "localhost", "port": 6383},
        {"host": "localhost", "port": 6384},
    ])
    
    # RabbitMQ cluster nodes
    rabbitmq_nodes: List[str] = field(default_factory=lambda: [
        "amqp://raguser:ragpass@localhost:5672/",
        "amqp://raguser:ragpass@localhost:5673/",
        "amqp://raguser:ragpass@localhost:5674/",
    ])
    
    # Collection/DB names
    qdrant_collection: str = "nodejs"
    mongodb_database: str = "rag_db"
    
    # Sharding config
    shard_count: int = 3
    replication_factor: int = 2
    
    # Embedding model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072


@dataclass
class RAGConfig:
    """Main configuration container"""
    mode: RAGMode
    single: SingleNodeConfig
    distributed: DistributedConfig
    
    @property
    def is_distributed(self) -> bool:
        return self.mode == RAGMode.DISTRIBUTED
    
    def get_qdrant_url(self) -> str:
        """Get primary Qdrant URL for current mode"""
        if self.is_distributed:
            return self.distributed.qdrant_nodes[0]
        return self.single.qdrant_url
    
    def get_mongodb_url(self) -> str:
        """Get MongoDB connection URL for current mode"""
        if self.is_distributed:
            return self.distributed.mongodb_uri
        return self.single.mongodb_url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL for current mode"""
        if self.is_distributed:
            # Return first node for cluster bootstrap
            node = self.distributed.redis_nodes[0]
            return f"redis://{node['host']}:{node['port']}"
        return self.single.redis_url
    
    def get_rabbitmq_url(self) -> str:
        """Get RabbitMQ URL for current mode"""
        if self.is_distributed:
            return self.distributed.rabbitmq_nodes[0]
        return self.single.rabbitmq_url
    
    def get_collection_name(self) -> str:
        """Get Qdrant collection name"""
        if self.is_distributed:
            return self.distributed.qdrant_collection
        return self.single.qdrant_collection


# Global config instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """
    Get the global configuration instance.
    
    Configuration is determined by RAG_MODE environment variable:
    - 'distributed' or 'cluster': Use distributed mode
    - Any other value or unset: Use single mode
    
    Returns:
        RAGConfig instance
    """
    global _config
    
    if _config is None:
        mode_str = os.getenv("RAG_MODE", "single").lower()
        
        if mode_str in ("distributed", "cluster"):
            mode = RAGMode.DISTRIBUTED
        else:
            mode = RAGMode.SINGLE
        
        _config = RAGConfig(
            mode=mode,
            single=SingleNodeConfig(),
            distributed=DistributedConfig(),
        )
        
        print(f"[Config] RAG Mode: {_config.mode.value}")
    
    return _config


def set_mode(mode: RAGMode) -> RAGConfig:
    """
    Override the configuration mode programmatically.
    
    Args:
        mode: The RAGMode to set
        
    Returns:
        Updated RAGConfig instance
    """
    global _config
    
    _config = RAGConfig(
        mode=mode,
        single=SingleNodeConfig(),
        distributed=DistributedConfig(),
    )
    
    print(f"[Config] RAG Mode set to: {_config.mode.value}")
    return _config


def reset_config():
    """Reset configuration to be re-initialized on next get_config() call"""
    global _config
    _config = None
