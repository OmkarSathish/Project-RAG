"""
Hybrid Retriever combining semantic and keyword search with Reciprocal Rank Fusion.

Educational concepts:

1. Semantic Search (Dense Vectors):
   - Uses embeddings to capture semantic meaning
   - Good for: Conceptual queries, synonyms, paraphrasing
   - Example: "How to start server" matches "launching application"

2. Keyword Search (BM25):
   - Traditional term-based matching with TF-IDF weighting
   - Good for: Exact terms, technical jargon, names
   - Example: "express.listen()" requires exact term match

3. Reciprocal Rank Fusion (RRF):
   - Combines rankings from multiple retrieval methods
   - Formula: score = sum(weight / (k + rank)) for each method
   - k=60 is empirically proven default
   - Robust to score scale differences

Hybrid retrieval typically improves recall by 15-30% over single method.
"""

from typing import List
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """Combines semantic (dense) and keyword (BM25) search with RRF fusion"""

    def __init__(self, vector_store, documents):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: LangChain vector store (for semantic search)
            documents: List of all documents (for BM25 indexing)
        """
        self.vector_store = vector_store
        self.documents = documents

        # Build BM25 index for keyword search
        print("[HybridRetriever] Building BM25 index...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"[HybridRetriever] BM25 index built with {len(documents)} documents")

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List:
        """
        Perform hybrid retrieval combining semantic and keyword search.

        Process:
        1. Semantic search: Retrieve top-K using vector similarity
        2. BM25 search: Retrieve top-K using keyword matching
        3. RRF fusion: Combine rankings with weighted scores
        4. Return merged and sorted results

        Args:
            query: User query string
            k: Number of candidates to retrieve from each method
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for BM25 search (0-1)

        Returns:
            List of documents sorted by hybrid relevance score
        """
        # 1. Semantic search with embeddings
        semantic_results = await self.vector_store.asimilarity_search_with_score(
            query=query, k=k * 2  # Retrieve more for better fusion
        )

        # 2. BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: k * 2]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_constant = 60  # Standard RRF constant

        # Add semantic search scores
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                semantic_weight / (k_constant + rank)
            )

        # Add BM25 scores
        for rank, idx in enumerate(bm25_top_indices, 1):
            doc = self.documents[idx]
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                keyword_weight / (k_constant + rank)
            )

        # 4. Sort by RRF score and return top-k
        sorted_doc_ids = sorted(
            [(doc_id, score) for doc_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        # Map doc IDs back to document objects
        doc_map = {id(doc): doc for doc, _ in semantic_results}
        doc_map.update(
            {id(self.documents[idx]): self.documents[idx] for idx in bm25_top_indices}
        )

        final_results = [doc_map[doc_id] for doc_id, _ in sorted_doc_ids]

        print(
            f"[HybridRetriever] Fused {len(semantic_results)} semantic + {len(bm25_top_indices)} BM25 -> {len(final_results)} results"
        )

        return final_results
