"""
Zep Memory Client
Manages long-term memory using Zep Cloud.
Keeps Voyage AI for embeddings.
"""

import logging
from typing import Optional, List
from datetime import datetime

from zep_cloud.client import Zep
from zep_cloud.types import Message
from zep_cloud.errors import NotFoundError

from .embedding import VoyageEmbedder
from .reranker import VoyageReranker

logger = logging.getLogger("memory_chat.zep")


class ZepMemoryClient:
    """
    Zep Cloud client for long-term memory.
    
    Zep handles:
    - Thread management: automatic creation
    - Message storage: thread.add_messages()
    - Context retrieval: thread.get_user_context()
    - Business data: graph.add()
    
    Voyage AI handles:
    - Custom embeddings for additional search
    - Reranking of results
    """
    
    def __init__(
        self,
        zep_api_key: str,
        voyage_api_key: Optional[str] = None,
        voyage_embed_model: str = "voyage-3-large",
        voyage_rerank_model: str = "rerank-2.5",
    ):
        """
        Initialize the Zep memory client.
        
        Args:
            zep_api_key: Zep Cloud API key
            voyage_api_key: Optional Voyage AI API key for embeddings
            voyage_embed_model: Voyage embedding model
            voyage_rerank_model: Voyage reranker model
        """
        logger.info("INITIALIZING ZEP MEMORY CLIENT")
        
        # Initialize Zep client
        self.zep = Zep(api_key=zep_api_key)        
        # Track created threads
        self._created_threads: set = set()
        
        # Initialize Voyage AI for embeddings (optional)
        self.voyage_embedder = None
        self.voyage_reranker = None
        if voyage_api_key:
            self.voyage_embedder = VoyageEmbedder(voyage_api_key, voyage_embed_model)
            self.voyage_reranker = VoyageReranker(voyage_api_key, voyage_rerank_model)
            logger.info(f"[INIT] Voyage AI initialized - embed: {voyage_embed_model}, rerank: {voyage_rerank_model}")
        
        logger.info("[INIT] ZepMemoryClient initialization complete")
    
    def _get_thread_id(self, user_id: str) -> str:
        """
        Get thread ID for a user.
        Uses user_id as thread_id for simplicity.
        """
        return f"thread_{user_id}"
    
    async def _ensure_thread_exists(self, user_id: str) -> str:
        """
        Ensure thread exists for user, create if not.
        
        Args:
            user_id: User identifier
        
        Returns:
            Thread ID
        """
        thread_id = self._get_thread_id(user_id)
        
        # Check if we've already created this thread
        if thread_id in self._created_threads:
            logger.debug(f"[THREAD] Thread {thread_id} already created (cached)")
            return thread_id
        
        try:
            # Try to get existing thread
            logger.info(f"[THREAD] Checking if thread exists: {thread_id}")
            self.zep.thread.get(thread_id=thread_id)
            self._created_threads.add(thread_id)
            logger.info(f"[THREAD] Thread exists: {thread_id}")
            return thread_id
            
        except NotFoundError:
            # User and thread don't exist - create user first, then thread
            logger.info(f"[THREAD] Thread not found, creating user and thread for: {user_id}")
            
            # First, try to create the user (Zep requires user to exist before thread)
            try:
                logger.info(f"[USER] Creating user: {user_id}")
                self.zep.user.add(user_id=user_id)
                logger.info(f"[USER] User created: {user_id}")
            except Exception as user_error:
                # User might already exist, that's okay
                logger.info(f"[USER] User creation result: {user_error}")
            
            # Now create the thread
            logger.info(f"[THREAD] Creating thread: {thread_id}")
            self.zep.thread.create(
                thread_id=thread_id,
                user_id=user_id,
            )
            self._created_threads.add(thread_id)
            logger.info(f"[THREAD] Created new thread: {thread_id} for user: {user_id}")
            return thread_id
            
        except Exception as e:
            logger.exception(f"[THREAD] Error ensuring thread exists: {e}")
            raise
    
    async def add_messages(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        return_context: bool = True
    ) -> Optional[str]:
        """
        Add conversation messages to Zep.
        
        Args:
            user_id: User identifier
            user_message: The user's message
            assistant_response: The assistant's response
            return_context: Return context block immediately
        
        Returns:
            Context block if return_context=True, else None
        """
        logger.info("[STORE] ADDING MESSAGES TO ZEP")
        
        try:
            # Ensure thread exists
            thread_id = await self._ensure_thread_exists(user_id)
            
            logger.info(f"[STORE] Thread ID: {thread_id}")
            logger.info(f"[STORE] User Message: {user_message}")
            logger.info(f"[STORE] Assistant Response: {assistant_response}")
            
            messages = [
                Message(
                    role="user",
                    content=user_message,
                ),
                Message(
                    role="assistant",
                    content=assistant_response,
                ),
            ]
            
            logger.info(f"[STORE] Sending {len(messages)} messages to Zep...")
            logger.debug(f"[STORE] Messages: {messages}")
            
            response = self.zep.thread.add_messages(
                thread_id=thread_id,
                messages=messages,
                return_context=return_context,
            )
            
            logger.info(f"[STORE] Messages added successfully!")
            
            if return_context and hasattr(response, 'context'):
                logger.info(f"[STORE] Received context block, length: {len(response.context)}")
                logger.debug(f"[STORE] Context preview: {response.context}")
                return response.context
            
            logger.info("[STORE] No context returned")
            return None
            
        except Exception as e:
            logger.exception(f"[STORE] Error adding messages: {e}")
            return None
    
    async def get_context(self, user_id: str) -> str:
        """
        Get user context from Zep.
        
        Returns a formatted context block with:
        - User summary
        - Relevant facts with date ranges
        
        Args:
            user_id: User identifier
        
        Returns:
            Formatted context string
        """
        logger.info("[RETRIEVE] GETTING CONTEXT FROM ZEP")
        
        try:
            thread_id = self._get_thread_id(user_id)
            
            logger.info(f"[RETRIEVE] User ID: {user_id}")
            logger.info(f"[RETRIEVE] Thread ID: {thread_id}")
            
            # Check if thread exists first
            if thread_id not in self._created_threads:
                try:
                    self.zep.thread.get(thread_id=thread_id)
                    self._created_threads.add(thread_id)
                except NotFoundError:
                    logger.info(f"[RETRIEVE] Thread does not exist yet: {thread_id}")
                    logger.info("[RETRIEVE] No context available (new user)")
                    return ""
            
            logger.info(f"[RETRIEVE] Fetching user context...")
            
            user_context = self.zep.thread.get_user_context(thread_id=thread_id)
            
            if user_context and hasattr(user_context, 'context') and user_context.context:
                context = user_context.context
                logger.info(f"[RETRIEVE] Context retrieved successfully!")
                logger.info(f"[RETRIEVE] Context length: {len(context)} characters")
                logger.info(f"[RETRIEVE] Context preview:\n{context}")
                return context
            
            logger.info("[RETRIEVE] No context found for user")
            return ""
            
        except NotFoundError:
            logger.info(f"[RETRIEVE] Thread not found: {thread_id}")
            return ""
        except Exception as e:
            logger.exception(f"[RETRIEVE] Error getting context: {e}")
            return ""
    
    async def add_business_data(
        self,
        user_id: str,
        data: str,
        data_type: str = "text"
    ) -> str:
        """
        Add business data (documents) to user's graph.
        
        Args:
            user_id: User identifier
            data: The data/document content
            data_type: Type of data
        
        Returns:
            Status message
        """
        logger.info("[DOCUMENT] ADDING BUSINESS DATA TO ZEP")
        
        try:
            logger.info(f"[DOCUMENT] User ID: {user_id}")
            logger.info(f"[DOCUMENT] Data Type: {data_type}")
            logger.info(f"[DOCUMENT] Data Length: {len(data)} characters")
            logger.info(f"[DOCUMENT] Data Preview: {data}")
            
            # Ensure user exists by creating a thread first
            await self._ensure_thread_exists(user_id)
            
            # Add data to graph - type must be "message", "text", or "json"
            self.zep.graph.add(
                user_id=user_id,
                type="text",  # Use "text" for document content
                data=data,
            )
            
            logger.info(f"[DOCUMENT] Data added successfully!")
            return "Data added successfully."
            
        except Exception as e:
            logger.exception(f"[DOCUMENT] Error adding data: {e}")
            return f"Error adding data: {str(e)}"
    
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        use_reranking: bool = True
    ) -> List[str]:
        """
        Search user's memory with optional Voyage AI reranking.
        
        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results
            use_reranking: Apply Voyage AI reranking
        
        Returns:
            List of relevant facts/memories
        """
        logger.info(f"[SEARCH] Searching for user: {user_id}, query: {query}")
        
        try:
            # Get context which includes relevant facts
            context = await self.get_context(user_id)
            
            if not context:
                logger.info("[SEARCH] No context to search")
                return []
            
            # Parse facts from context block
            facts = self._parse_facts(context)
            logger.info(f"[SEARCH] Parsed {len(facts)} facts from context")
            
            # Apply Voyage AI reranking if available
            if use_reranking and self.voyage_reranker and len(facts) > 1:
                logger.info(f"[SEARCH] Reranking {len(facts)} results with Voyage AI")
                reranked = await self.voyage_reranker.rerank(query, facts, top_k=limit)
                facts = [r.document for r in reranked]
            
            return facts[:limit]
            
        except Exception as e:
            logger.exception(f"[SEARCH] Error: {e}")
            return []
    
    def _parse_facts(self, context: str) -> List[str]:
        """
        Parse facts from Zep context block.
        
        Args:
            context: Raw context string from Zep
        
        Returns:
            List of fact strings
        """
        facts = []
        
        # Look for facts between <FACTS> tags
        if "<FACTS>" in context and "</FACTS>" in context:
            facts_section = context.split("<FACTS>")[1].split("</FACTS>")[0]
            for line in facts_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    # Remove the leading "- " and any date range
                    fact = line[2:].strip()
                    if "(" in fact:
                        fact = fact.split("(")[0].strip()
                    if fact:
                        facts.append(fact)
        
        logger.debug(f"[PARSE] Extracted {len(facts)} facts from context")
        return facts
    
    async def close(self) -> None:
        """Close the Zep client connection."""
        logger.info("ZepMemoryClient closed")
