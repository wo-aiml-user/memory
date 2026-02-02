"""
Supermemory Client
Manages long-term memory using Supermemory.ai.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any

from supermemory import Supermemory

# Configure logging
logger = logging.getLogger("memory_chat.supermemory")

class SupermemoryClient:
    """
    Supermemory client for long-term memory and document storage.
    
    Replaces ZepMemoryClient.
    
    Features:
    - Ingest text/conversations with `add()`
    - Semantic search with `search.memories()`
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Supermemory client.
        
        Args:
            api_key: Supermemory API key
        """
        logger.info("INITIALIZING SUPERMEMORY CLIENT")
        if not api_key:
            logger.warning("Supermemory API Key is missing!")
            
        self.client = Supermemory(api_key=api_key) if api_key else None
        
        logger.info("[INIT] Supermemory initialization complete")

    async def add_messages(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        return_context: bool = False # Deprecated but kept for signature compatibility if needed
    ) -> None:
        """
        Add conversation pair to Supermemory.
        
        Args:
            user_id: User identifier (used as container_tag)
            user_message: User input
            assistant_response: AI response
        """
        if not self.client:
            logger.warning("[STORE] Client not initialized, skipping storage")
            return

        logger.info(f"[STORE] Adding conversation to Supermemory for user: {user_id}")
        
        try:
            # Format as a conversation block
            content = f"user: {user_message}\nassistant: {assistant_response}"
            
            # We assume user_id is the container_tag
            # Custom ID can be random for each turn, or linked if we had a conversation ID
            # For now, just ingestion is key.
            
            response = self.client.add(
                content=content,
                container_tag=user_id,
                metadata={"type": "conversation"}
            )
            
            logger.info(f"[STORE] [OK] Success: {response}")
            
        except Exception as e:
            logger.exception(f"[STORE] Error adding messages: {e}")

    async def search_context(self, user_id: str, query: str, limit: int = 5) -> str:
        """
        Search for relevant context based on a query.
        
        Args:
            user_id: User identifier (container_tag)
            query: The search query (usually the user's last message)
            limit: Max results
            
        Returns:
            Formatted string of relevant memories/chunks
        """
        if not self.client:
            return ""

        logger.info(f"[SEARCH] Searching Supermemory: query='{query}' user='{user_id}'")
        
        try:
            # Hybrid search is recommended
            response = self.client.search.memories(
                q=query,
                container_tag=user_id,
                search_mode="hybrid",
                limit=limit
            )
            
            # Determine if we have a list of objects or dicts (though error suggests objects)
            results = getattr(response, "results", [])
            
            if not results:
                logger.info("[SEARCH] No relevant context found.")
                return ""
                
            formatted_context = []
            for item in results:
                # Handle both 'memory' (facts) and 'chunk' (document snippets)
                # item is main object in result list
                text = getattr(item, "memory", None) or getattr(item, "chunk", None) or ""
                similarity = getattr(item, "similarity", 0)
                
                if text:
                    formatted_context.append(f"- {text} (similarity: {similarity:.2f})")
            
            context_str = "\n".join(formatted_context)
            logger.info(f"[SEARCH] Found {len(results)} relevant items.")
            return context_str
            
        except Exception as e:
            logger.exception(f"[SEARCH] Error retrieving context: {e}")
            return ""

    async def add_business_data(
        self,
        user_id: str,
        data: str,
        source: str = "document"
    ) -> str:
        """
        Add a document to Supermemory.
        
        Supermemory handles chunking automatically.
        
        Args:
            user_id: User identifier
            data: Document text content
            source: Source identifier
            
        Returns:
            Status message
        """
        if not self.client:
            return "Client not initialized"

        logger.info(f"[DOCUMENT] Adding document to Supermemory. Size: {len(data)} chars")
        
        try:
            # Supermemory handles large text automatically
            response = self.client.add(
                content=data,
                container_tag=user_id,
                metadata={"source": source, "type": "document"}
            )
            
            # Fallback to getattr just in case response is object
            doc_id = getattr(response, "id", "unknown")
            status = f"Document stored successfully. ID: {doc_id}"
            logger.info(f"[DOCUMENT] [OK] {status}")
            return status
            
        except Exception as e:
            logger.exception(f"[DOCUMENT] Error adding document: {e}")
            return f"Error adding document: {str(e)}"

    async def close(self) -> None:
        """No-op for HTTP client usually, but kept for interface compatibility."""
        pass
