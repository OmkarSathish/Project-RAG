"""
Core RAG Engine - Wraps the LangGraph RAG chain for use in web applications.

This module provides a clean interface to the RAG system for use by the FastAPI application.
"""

from reranker import Reranker
import asyncio
import hashlib
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, END
import redis.asyncio as redis
import sys
from pathlib import Path

# Add project root to path to import reranker
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class State(TypedDict):
    messages: List[dict]
    context: str
    relevance_score: int | None
    run_count: int
    cache_hit: bool


class RAGEngine:
    """Core RAG engine for processing queries"""

    def __init__(self):
        self.llm = AsyncOpenAI()
        self.redis_client: Optional[redis.Redis] = None
        self.reranker: Optional[Reranker] = None

        # Initialize embedding model and vector store
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="nodejs",
            embedding=self.embedding_model,
            force_disable_check_same_thread=True,
        )

        # Build LangGraph
        self.graph = self._build_graph()

    async def get_redis_client(self):
        """Initialize Redis client lazily"""
        if self.redis_client is None:
            self.redis_client = await redis.from_url("redis://localhost:6379")
        return self.redis_client

    def get_reranker(self):
        """Initialize reranker lazily"""
        if self.reranker is None:
            self.reranker = Reranker()
        return self.reranker

    async def search_node(self, state: State):
        """Search node with caching and reranking"""
        user_query = state["messages"][-1]["content"]

        # Generate cache key
        cache_key = f"search:{hashlib.md5(user_query.encode()).hexdigest()}"

        # Try cache
        redis_conn = await self.get_redis_client()
        cached_context = await redis_conn.get(cache_key)

        if cached_context:
            state["context"] = cached_context.decode("utf-8")
            state["cache_hit"] = True
        else:
            # Perform search with reranking
            initial_results = await self.vector_db.asimilarity_search(
                query=user_query, k=10
            )

            # Rerank
            reranker_model = self.get_reranker()
            reranked_results = await asyncio.to_thread(
                reranker_model.rerank, user_query, initial_results, top_k=4
            )

            # Format context - handle both LangChain and direct Qdrant payload formats
            context = "\n\n".join(
                [
                    f"Page Content: {result.page_content}\n Page Number: {result.metadata.get('page_label', result.metadata.get('page', 'N/A'))}\nFile Location:{result.metadata.get('source', 'Unknown')}"
                    for result in reranked_results
                ]
            )

            # Cache for 5 minutes
            await redis_conn.setex(cache_key, 300, context)
            state["context"] = context
            state["cache_hit"] = False

        return state

    async def llm_node(self, state: State):
        """LLM generation node"""
        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the available context retrieved from a PDF file
        along with page contents and page number.
        You should only answer the user based on the following context and navigate the user to open the right page number
        to know more.
        Context:
        {state["context"]}
        """
        state["messages"].append({"role": "system", "content": SYSTEM_PROMPT})
        llm_response = await self.llm.chat.completions.create(
            model="gpt-4o-mini", messages=state["messages"]
        )
        system_msg = {
            "role": "system",
            "content": llm_response.choices[0].message.content,
        }
        state["messages"].append(system_msg)
        return state

    def validate_conditional_edge(self, state: State):
        """Simple conditional - skip validation for demo"""
        return END

    def _build_graph(self):
        """Build the LangGraph state machine"""
        graph = StateGraph(State)
        graph.add_node("search", self.search_node)
        graph.add_node("llm", self.llm_node)
        graph.add_edge("search", "llm")
        graph.add_edge("llm", END)
        graph.set_entry_point("search")
        return graph

    async def process_query(
        self, query: str, client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query string
            client_id: Unique client identifier

        Returns:
            Dict with response, cache_hit status, and timing info
        """
        import time

        start_time = time.time()

        # Create state
        user_msg = {"role": "user", "content": query}
        rag_state: State = {
            "messages": [user_msg],
            "context": "",
            "run_count": 0,
            "relevance_score": None,
            "cache_hit": False,
        }

        # Use simple in-memory checkpointer for demo
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        rag_chain = self.graph.compile(checkpointer=checkpointer)

        # Process
        config = {"configurable": {"thread_id": client_id}}
        final_state = await rag_chain.ainvoke(rag_state, config)

        # Extract response
        response_content = final_state["messages"][-1]["content"]
        cache_hit = final_state.get("cache_hit", False)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            "response": response_content,
            "cache_hit": cache_hit,
            "processing_time": round(processing_time, 2),
            "client_id": client_id,
        }
