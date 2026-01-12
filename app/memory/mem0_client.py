"""
Mem0 Memory Client
Manages long-term memory using Mem0 Platform.
"""

import logging
from typing import Optional, List, Dict, Any

from mem0 import MemoryClient

logger = logging.getLogger("memory_chat.mem0")


class Mem0MemoryClient:
    """
    Mem0 Platform client for long-term memory.
    
    Mem0 handles:
    - Memory storage: client.add()
    - Memory search: client.search()
    - Memory retrieval: client.get_all()
    - Automatic fact extraction with infer=True
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Mem0 memory client.
        
        Args:
            api_key: Mem0 Platform API key
        """
        logger.info("=" * 70)
        logger.info("[MEM0] INITIALIZING MEM0 MEMORY CLIENT")
        logger.info("=" * 70)
        
        self.client = MemoryClient(api_key=api_key)
        
        logger.info("[MEM0] Mem0MemoryClient initialized successfully")
    
    async def add_messages(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Add conversation messages to Mem0.
        
        Args:
            user_id: User identifier
            user_message: The user's message
            assistant_response: The assistant's response
        
        Returns:
            Result from Mem0 add operation
        """
        logger.info("=" * 70)
        logger.info("[MEM0 STORE] ADDING MESSAGES TO MEM0")
        logger.info("=" * 70)
        
        try:
            logger.info(f"[MEM0 STORE] User ID: {user_id}")
            
            # Log full message content
            logger.info("-" * 50)
            logger.info("[MEM0 STORE] MESSAGE PAYLOAD TO MEM0:")
            logger.info("-" * 50)
            logger.info("[MEM0 STORE] Message 1 (role=user):")
            for line in user_message.split('\n'):
                logger.info(f"  | {line}")
            logger.info("[MEM0 STORE] Message 2 (role=assistant):")
            for line in assistant_response.split('\n'):
                logger.info(f"  | {line}")
            logger.info("-" * 50)
            
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
            
            logger.info(f"[MEM0 STORE] Sending {len(messages)} messages to Mem0 API...")
            logger.info(f"[MEM0 STORE] User message length: {len(user_message)} chars")
            logger.info(f"[MEM0 STORE] Assistant response length: {len(assistant_response)} chars")
            
            # Add to Mem0 with automatic fact extraction (infer=True by default)
            result = self.client.add(
                messages=messages,
                user_id=user_id,
            )
            
            logger.info(f"[MEM0 STORE] [OK] Messages added to Mem0 successfully!")
            logger.info(f"[MEM0 STORE] Mem0 will extract facts and entities automatically")
            
            if result:
                logger.info(f"[MEM0 STORE] Result: {result}")
            
            return result
            
        except Exception as e:
            logger.exception(f"[MEM0 STORE] Error adding messages: {e}")
            return None
    
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search user's memory for relevant information.
        
        Args:
            user_id: User identifier
            query: Search query (natural language)
            limit: Maximum number of results
        
        Returns:
            List of relevant memories
        """
        logger.info("=" * 70)
        logger.info("[MEM0 SEARCH] SEARCHING MEMORIES IN MEM0")
        logger.info("=" * 70)
        
        try:
            logger.info(f"[MEM0 SEARCH] User ID: {user_id}")
            logger.info(f"[MEM0 SEARCH] Query: {query}")
            logger.info(f"[MEM0 SEARCH] Limit: {limit}")
            
            logger.info(f"[MEM0 SEARCH] Calling Mem0 API: search()...")
            
            # Search with user_id filter
            results = self.client.search(
                query=query,
                user_id=user_id,
                limit=limit,
            )
            
            if results:
                logger.info(f"[MEM0 SEARCH] [OK] Found {len(results)} memories!")
                logger.info("-" * 50)
                logger.info("[MEM0 SEARCH] SEARCH RESULTS:")
                logger.info("-" * 50)
                for i, memory in enumerate(results, 1):
                    memory_text = memory.get('memory', str(memory))
                    logger.info(f"  [{i}] {memory_text}")
                logger.info("-" * 50)
            else:
                logger.info("[MEM0 SEARCH] No memories found for query")
            
            return results if results else []
            
        except Exception as e:
            logger.exception(f"[MEM0 SEARCH] Error searching memories: {e}")
            return []
    
    async def get_context(self, user_id: str, query: str) -> str:
        """
        Get formatted context string from user's memories.
        
        This is the main method used by MemoryChain to inject context.
        Uses the user's message as the search query.
        
        Args:
            user_id: User identifier
            query: The user's current message (used as search query)
        
        Returns:
            Formatted context string for LLM injection
        """
        logger.info("=" * 70)
        logger.info("[MEM0 CONTEXT] GETTING CONTEXT FROM MEM0")
        logger.info("=" * 70)
        
        try:
            logger.info(f"[MEM0 CONTEXT] User ID: {user_id}")
            logger.info(f"[MEM0 CONTEXT] Search Query: {query}")
            
            # Search for relevant memories
            memories = await self.search(user_id, query, limit=10)
            
            if not memories:
                logger.info("[MEM0 CONTEXT] No memories found - this may be a new user")
                return ""
            
            # Format memories into context block
            context_lines = ["<MEMORIES>"]
            for memory in memories:
                memory_text = memory.get('memory', str(memory))
                context_lines.append(f"- {memory_text}")
            context_lines.append("</MEMORIES>")
            
            context = "\n".join(context_lines)
            
            logger.info(f"[MEM0 CONTEXT] [OK] Context built from {len(memories)} memories")
            logger.info(f"[MEM0 CONTEXT] Context length: {len(context)} characters")
            logger.info("-" * 50)
            logger.info("[MEM0 CONTEXT] FULL CONTEXT:")
            logger.info("-" * 50)
            for line in context.split('\n'):
                logger.info(f"  | {line}")
            logger.info("-" * 50)
            
            return context
            
        except Exception as e:
            logger.exception(f"[MEM0 CONTEXT] Error getting context: {e}")
            return ""
    
    async def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of all user memories
        """
        logger.info(f"[MEM0] Getting all memories for user: {user_id}")
        
        try:
            memories = self.client.get_all(user_id=user_id)
            logger.info(f"[MEM0] Retrieved {len(memories) if memories else 0} total memories")
            return memories if memories else []
            
        except Exception as e:
            logger.exception(f"[MEM0] Error getting all memories: {e}")
            return []
    
    async def close(self) -> None:
        """Close the Mem0 client (no-op, kept for interface compatibility)."""
        logger.info("[MEM0] Mem0MemoryClient closed")
