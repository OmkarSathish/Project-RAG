"""
RAG Engine Factory - Unified interface for single and distributed modes.

This module provides a factory pattern to instantiate the correct RAG engine
based on the configured mode (single or distributed).

Usage:
    from src.core.engine_factory import create_rag_engine, get_rag_engine
    
    # Create based on environment (RAG_MODE env var)
    engine = await create_rag_engine()
    
    # Or explicitly specify mode
    engine = await create_rag_engine(mode="distributed")
"""

from typing import Optional, Any
from .config import get_config, set_mode, RAGMode


# Type alias - using Any to avoid circular imports
RAGEngineType = Any

# Global engine instance
_engine: Optional[RAGEngineType] = None


async def create_rag_engine(mode: Optional[str] = None) -> RAGEngineType:
    """
    Create and initialize a RAG engine based on the specified mode.
    
    Args:
        mode: Optional mode override ('single' or 'distributed').
              If not specified, uses RAG_MODE environment variable.
    
    Returns:
        Initialized RAG engine (either RAGEngine or DistributedRAGEngine)
    """
    global _engine
    
    # Get or set config based on mode
    if mode:
        config = set_mode(RAGMode.DISTRIBUTED if mode == "distributed" else RAGMode.SINGLE)
    else:
        config = get_config()
    
    if config.is_distributed:
        # Import and create distributed engine
        from .rag_engine_distributed import DistributedRAGEngine
        from cluster_clients import ClusterConfig
        
        # Create cluster config from RAG config
        cluster_config = ClusterConfig(
            qdrant_nodes=config.distributed.qdrant_nodes,
            mongodb_uri=config.distributed.mongodb_uri,
            redis_nodes=config.distributed.redis_nodes,
            rabbitmq_nodes=config.distributed.rabbitmq_nodes,
        )
        
        engine = DistributedRAGEngine(config=cluster_config)
        await engine.initialize()
        
    else:
        # Import and create single-node engine
        from .rag_engine import RAGEngine
        engine = RAGEngine()
    
    _engine = engine
    return engine


async def get_rag_engine() -> RAGEngineType:
    """
    Get the current RAG engine instance.
    
    Creates a new engine if one doesn't exist.
    
    Returns:
        The active RAG engine instance
    """
    global _engine
    
    if _engine is None:
        _engine = await create_rag_engine()
    
    return _engine


async def shutdown_engine():
    """
    Shutdown the current RAG engine and clean up resources.
    """
    global _engine
    
    if _engine is None:
        return
    
    config = get_config()
    
    if config.is_distributed:
        # Close distributed client connections
        if hasattr(_engine, 'clients') and _engine.clients:
            await _engine.clients.close_all()
    
    _engine = None
    print("[EngineFactory] Engine shutdown complete")


def get_engine_info() -> dict:
    """
    Get information about the current engine configuration.
    
    Returns:
        Dictionary with mode and connection details
    """
    config = get_config()
    
    info = {
        "mode": config.mode.value,
        "is_distributed": config.is_distributed,
    }
    
    if config.is_distributed:
        info["qdrant_nodes"] = config.distributed.qdrant_nodes
        info["mongodb_uri"] = config.distributed.mongodb_uri
        info["redis_node_count"] = len(config.distributed.redis_nodes)
        info["rabbitmq_node_count"] = len(config.distributed.rabbitmq_nodes)
    else:
        info["qdrant_url"] = config.single.qdrant_url
        info["mongodb_url"] = config.single.mongodb_url
        info["redis_url"] = config.single.redis_url
    
    return info
