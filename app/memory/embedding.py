"""
Voyage AI Embedding Client
Provides text embedding functionality using Voyage AI API.
"""

import logging
from typing import Optional
import voyageai

logger = logging.getLogger("memory_chat.embedding")


class VoyageEmbedder:
    """Voyage AI embedding client for generating text embeddings."""
    
    def __init__(self, api_key: str, model: str = "voyage-3-large"):
        """
        Initialize the Voyage AI embedder.
        
        Args:
            api_key: Voyage AI API key
            model: Embedding model to use (default: voyage-3-large)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        logger.info(f"Initialized VoyageEmbedder with model: {model}")
    
    async def embed(
        self,
        texts: list[str], 
        input_type: Optional[str] = "document"
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            input_type: Type of input - "query" or "document" (default: document)
        
        Returns:
            List of embedding vectors
        """
        try:
            logger.debug(f"Embedding {len(texts)} texts with input_type={input_type}")
            
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type,
                truncation=True
            )
            
            logger.debug(f"Successfully generated {len(result.embeddings)} embeddings")
            return result.embeddings
            
        except Exception as e:
            logger.exception(f"Error generating embeddings: {e}")
            raise
    
    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string to embed
        
        Returns:
            Embedding vector
        """
        embeddings = await self.embed([query], input_type="query")
        return embeddings[0]
    
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of document strings to embed
        
        Returns:
            List of embedding vectors
        """
        return await self.embed(documents, input_type="document")
