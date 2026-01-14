"""
Core RAG Engine with Distributed Database Support.

This module provides a production-grade RAG engine using:
- Qdrant cluster for vector search
- Redis cluster for caching
- MongoDB replica set for state persistence
"""

import asyncio
import hashlib
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from reranker import Reranker
from cluster_clients import (
    DistributedClientManager, 
    ClusterConfig,
)


class State(TypedDict):
    messages: List[dict]
    context: str
    relevance_score: int | None
    run_count: int
    cache_hit: bool


class DistributedRAGEngine:
    """
    Production RAG engine with distributed database support.
    
    Features:
    - Cluster-aware connections with automatic failover
    - Distributed caching across Redis cluster
    - Vector search across Qdrant shards
    - State persistence in MongoDB replica set
    """
    
    def __init__(self, config: ClusterConfig = None):
        """
        Initialize distributed RAG engine.
        
        Args:
            config: Cluster configuration (uses defaults if not provided)
        """
        self.config = config or ClusterConfig()
        self.llm = AsyncOpenAI()
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.reranker: Optional[Reranker] = None
        self.clients: Optional[DistributedClientManager] = None
        self.graph = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all cluster connections"""
        if self._initialized:
            return
        
        print("[DistributedRAGEngine] Initializing...")
        
        # Connect to all clusters
        self.clients = DistributedClientManager(self.config)
        await self.clients.connect_all()
        
        # Build LangGraph
        self.graph = self._build_graph()
        
        self._initialized = True
        print("[DistributedRAGEngine] Ready!")
    
    def get_reranker(self) -> Reranker:
        """Lazy load reranker model"""
        if self.reranker is None:
            self.reranker = Reranker()
        return self.reranker
    
    async def search_node(self, state: State) -> State:
        """
        Search node with distributed caching and vector search.
        
        Uses Redis cluster for caching and Qdrant cluster for search.
        """
        user_query = state["messages"][-1]["content"]
        
        # Generate cache key
        cache_key = f"search:{hashlib.md5(user_query.encode()).hexdigest()}"
        
        # Try Redis cluster cache
        cached_context = await self.clients.redis.get(cache_key)
        
        if cached_context:
            state["context"] = cached_context
            state["cache_hit"] = True
            print("[Cache] HIT - Using cached results from Redis cluster")
        else:
            print("[Cache] MISS - Performing distributed vector search")
            
            # Generate query embedding
            query_embedding = await self.embedding_model.aembed_query(user_query)
            
            # Search across Qdrant cluster
            qdrant_client = await self.clients.qdrant.get_client()
            initial_results = await qdrant_client.search(
                collection_name="nodejs",
                query_vector=query_embedding,
                limit=10,
                with_payload=True,
            )
            
            # Convert to document format for reranker
            class MockDocument:
                def __init__(self, content, metadata):
                    self.page_content = content
                    self.metadata = metadata
            
            documents = [
                MockDocument(
                    result.payload.get("page_content", ""),
                    {
                        "page_label": result.payload.get("page_label", ""),
                        "source": result.payload.get("source", ""),
                    }
                )
                for result in initial_results
            ]
            
            # Rerank results
            reranker = self.get_reranker()
            reranked = await asyncio.to_thread(
                reranker.rerank, user_query, documents, top_k=4
            )
            
            # Format context
            context = "\n\n".join([
                f"Page Content: {doc.page_content}\n"
                f"Page Number: {doc.metadata['page_label']}\n"
                f"File Location: {doc.metadata['source']}"
                for doc in reranked
            ])
            
            # Cache in Redis cluster (5 minutes)
            await self.clients.redis.setex(cache_key, 300, context)
            
            state["context"] = context
            state["cache_hit"] = False
        
        return state
    
    async def llm_node(self, state: State) -> State:
        """LLM generation node"""
        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the 
        available context retrieved from a PDF file along with page contents 
        and page number. You should only answer based on the following context 
        and navigate the user to the right page to learn more.
        
        Context:
        {state["context"]}
        """
        
        state["messages"].append({"role": "system", "content": SYSTEM_PROMPT})
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=state["messages"],
        )
        
        assistant_msg = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        state["messages"].append(assistant_msg)
        
        return state
    
    async def validate_node(self, state: State) -> State:
        """Validate answer relevance"""
        VALIDATE_PROMPT = f"""
        Analyze the relevance between the user query and LLM response.
        Grade from 0-10 (0 = no relevance, 10 = perfect relevance).
        
        User query: {state["messages"][-3]["content"]}
        LLM response: {state["messages"][-1]["content"]}
        
        Respond with only a single digit (0-10).
        """
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": VALIDATE_PROMPT}],
        )
        
        try:
            score = int(response.choices[0].message.content.strip())
        except ValueError:
            score = 5
        
        state["relevance_score"] = score
        state["run_count"] += 1
        
        return state
    
    def validate_conditional_edge(self, state: State):
        """Determine next step based on validation"""
        if state["run_count"] >= 3:
            return END
        if state["relevance_score"] < 5:
            return "search"
        return END
    
    def _build_graph(self):
        """Build LangGraph state machine"""
        graph = StateGraph(State)
        graph.add_node("search", self.search_node)
        graph.add_node("llm", self.llm_node)
        graph.add_node("validate", self.validate_node)
        graph.add_edge("search", "llm")
        graph.add_edge("llm", "validate")
        graph.add_conditional_edges("validate", self.validate_conditional_edge)
        graph.set_entry_point("search")
        return graph
    
    async def process_query(
        self, 
        query: str, 
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a query through the distributed RAG pipeline.
        
        Args:
            query: User query string
            client_id: Unique client identifier
            
        Returns:
            Dict with response, cache status, and metrics
        """
        if not self._initialized:
            await self.initialize()
        
        import time
        start_time = time.time()
        
        # Initialize state
        initial_state: State = {
            "messages": [{"role": "user", "content": query}],
            "context": "",
            "run_count": 0,
            "relevance_score": None,
            "cache_hit": False,
        }
        
        # Run graph
        compiled_graph = self.graph.compile()
        final_state = await compiled_graph.ainvoke(initial_state)
        
        processing_time = time.time() - start_time
        
        return {
            "response": final_state["messages"][-1]["content"],
            "cache_hit": final_state["cache_hit"],
            "relevance_score": final_state["relevance_score"],
            "run_count": final_state["run_count"],
            "processing_time": round(processing_time, 2),
            "client_id": client_id,
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all cluster components"""
        if not self._initialized:
            return {"initialized": False}
        
        return await self.clients.health_check()
    
    async def close(self):
        """Clean shutdown of all connections"""
        if self.clients:
            await self.clients.close_all()
        self._initialized = False
        print("[DistributedRAGEngine] Shutdown complete")


# Singleton instance
_engine: Optional[DistributedRAGEngine] = None


async def get_distributed_engine() -> DistributedRAGEngine:
    """Get or create the distributed RAG engine singleton"""
    global _engine
    if _engine is None:
        _engine = DistributedRAGEngine()
        await _engine.initialize()
    return _engine
