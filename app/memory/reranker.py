"""
Voyage AI Reranker Client
Provides document reranking functionality using Voyage AI API.
"""

import logging
from typing import Optional
from dataclasses import dataclass
import voyageai

logger = logging.getLogger("memory_chat.reranker")


@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int
    document: str
    relevance_score: float


class VoyageReranker:
    """Voyage AI reranker client for reranking documents."""
    
    def __init__(self, api_key: str, model: str = "rerank-2.5"):
        """
        Initialize the Voyage AI reranker.
        
        Args:
            api_key: Voyage AI API key
            model: Reranker model to use (default: rerank-2.5)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        logger.info(f"Initialized VoyageReranker with model: {model}")
    
    async def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: Optional[int] = 5
    ) -> list[RerankResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of RerankResult sorted by relevance
        """
        try:
            if not documents:
                logger.debug("No documents to rerank")
                return []
            
            logger.debug(f"Reranking {len(documents)} documents for query: {query}")
            
            result = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=top_k,
                truncation=True
            )
            
            rerank_results = [
                RerankResult(
                    index=r.index,
                    document=r.document,
                    relevance_score=r.relevance_score
                )
                for r in result.results
            ]
            
            logger.debug(f"Reranked to {len(rerank_results)} results")
            return rerank_results
            
        except Exception as e:
            logger.exception(f"Error reranking documents: {e}")
            raise
