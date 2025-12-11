"""
Query Enhancement Techniques for improving RAG retrieval.

Educational concepts:

1. HyDE (Hypothetical Document Embeddings):
   - Problem: Queries and documents live in different semantic spaces
   - Solution: Generate hypothetical answer, embed it instead of query
   - Example: Query "how to start server?" -> Generate "To start the server, use..."
   - Improvement: 10-25% better retrieval accuracy

2. Query Expansion:
   - Generate multiple phrasings of the same question
   - Increases recall by capturing different ways to express intent
   - Example: "start server" -> ["launch server", "run server", "initialize server"]

3. Query Decomposition:
   - Break complex multi-part questions into simpler sub-questions
   - Enables better retrieval for complex queries
   - Example: "How to start server and configure SSL?" ->
     ["How to start server?", "How to configure SSL?"]
"""

from openai import AsyncOpenAI
from typing import List


class QueryEnhancer:
    """Advanced query transformation techniques for RAG"""

    def __init__(self, llm_client: AsyncOpenAI):
        """
        Initialize query enhancer.

        Args:
            llm_client: AsyncOpenAI client for query transformations
        """
        self.llm = llm_client

    async def apply_hyde(self, query: str) -> str:
        """
        Apply HyDE (Hypothetical Document Embeddings).

        Instead of searching with the query, generate a hypothetical answer
        and search with that. This bridges the semantic gap between queries
        (question form) and documents (answer form).

        Args:
            query: Original user query

        Returns:
            Hypothetical document/answer text
        """
        hyde_prompt = f"""Generate a detailed, technical answer to this question as if it appeared in documentation:

Question: {query}

Answer (be specific and technical):"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": hyde_prompt}],
            temperature=0.7,
            max_tokens=200,
        )

        hypothetical_doc = response.choices[0].message.content
        print(f"[HyDE] Generated hypothetical document ({len(hypothetical_doc)} chars)")
        return hypothetical_doc

    async def expand_query(self, query: str) -> List[str]:
        """
        Query expansion: Generate multiple alternative phrasings.

        This increases recall by searching for the same concept
        expressed in different ways.

        Args:
            query: Original user query

        Returns:
            List of alternative query phrasings (includes original)
        """
        expansion_prompt = f"""Generate 3 alternative phrasings of this question that preserve the same meaning:

Original: {query}

Alternatives (one per line):"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.8,
            max_tokens=150,
        )

        alternatives = response.choices[0].message.content.strip().split("\n")
        expanded_queries = [query] + [
            alt.strip("123456789.- ") for alt in alternatives if alt.strip()
        ]

        print(f"[QueryExpansion] Expanded to {len(expanded_queries)} variations")
        return expanded_queries

    async def decompose_query(self, query: str) -> List[str]:
        """
        Query decomposition: Break complex queries into sub-questions.

        This handles multi-hop reasoning by breaking down complex
        information needs into simpler components.

        Args:
            query: Complex user query

        Returns:
            List of simpler sub-questions
        """
        decompose_prompt = f"""Break down this complex question into 2-3 simpler sub-questions:

Complex Question: {query}

Sub-questions (one per line):"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": decompose_prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        sub_queries = response.choices[0].message.content.strip().split("\n")
        decomposed = [sq.strip("123456789.- ") for sq in sub_queries if sq.strip()]

        print(f"[QueryDecomposition] Decomposed into {len(decomposed)} sub-questions")
        return decomposed

    async def enhance_on_retry(
        self, original_query: str, previous_context: str, validation_feedback: str = ""
    ) -> str:
        """
        Enhance query for retry based on validation feedback.

        When initial retrieval fails validation, reformulate the query
        to focus on missing information.

        Args:
            original_query: Original user query
            previous_context: Context retrieved in previous attempt
            validation_feedback: Feedback from validator (optional)

        Returns:
            Enhanced query for retry
        """
        retry_prompt = f"""The initial search didn't find good results. Reformulate this query to find better information.

Original Query: {original_query}
Previous Context Preview: {previous_context[:300]}...
{f"Validation Feedback: {validation_feedback}" if validation_feedback else ""}

Reformulated Query (focus on missing information):"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0.5,
            max_tokens=100,
        )

        enhanced_query = response.choices[0].message.content.strip()
        print(f"[QueryEnhancement] Enhanced query for retry: {enhanced_query}")
        return enhanced_query
