"""
Cross-Encoder Reranker for improving retrieval accuracy.

Educational concepts:
- Bi-encoders (used in initial retrieval): Fast but less accurate
  - Encodes query and documents separately
  - Compares via cosine similarity

- Cross-encoders (used in reranking): Slower but more accurate
  - Processes query-document pairs together
  - Captures fine-grained interactions
  - Better for final ranking of top candidates

Typical pipeline:
1. Bi-encoder retrieves top-K candidates (fast, K=10-100)
2. Cross-encoder reranks to top-N results (accurate, N=3-5)
"""

from sentence_transformers import CrossEncoder
from typing import List


class Reranker:
    """Cross-encoder based reranker for improving retrieval accuracy"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder model.

        Args:
            model_name: HuggingFace model identifier
                - ms-marco-MiniLM-L-6-v2: Fast, good for general queries
                - ms-marco-TinyBERT-L-2: Fastest, lower accuracy
                - ms-marco-MiniLM-L-12: More accurate, slower
        """
        print(f"[Reranker] Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Model loaded successfully")

    def rerank(self, query: str, documents: List, top_k: int = 4) -> List:
        """
        Rerank documents using cross-encoder scoring.

        Process:
        1. Create query-document pairs
        2. Score each pair with cross-encoder (0-1 score)
        3. Sort by score descending
        4. Return top-k documents

        Args:
            query: User query string
            documents: List of Document objects from retrieval
            top_k: Number of top documents to return after reranking

        Returns:
            List of reranked Document objects (top-k)
        """
        if not documents:
            return []

        # Create [query, document_text] pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs - cross-encoder returns relevance scores
        # Higher score = more relevant
        scores = self.model.predict(pairs)

        # Combine documents with their scores
        scored_docs = list(zip(documents, scores))

        # Sort by score descending (highest relevance first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k documents (without scores)
        reranked = [doc for doc, score in scored_docs[:top_k]]

        print(
            f"[Reranker] Reranked {len(documents)} -> {len(reranked)} documents"
        )
        print(
            f"[Reranker] Top scores: {[round(score, 3) for _, score in scored_docs[:top_k]]}"
        )

        return reranked
