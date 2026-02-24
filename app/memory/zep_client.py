"""
Zep Memory Client
Manages long-term memory using Zep Cloud.
Keeps Voyage AI for embeddings.
"""

import logging
import json
import uuid
from typing import Optional, List
from datetime import datetime

from zep_cloud.client import Zep
from zep_cloud.types import Message
from zep_cloud.errors import NotFoundError

from .embedding import VoyageEmbedder
from .reranker import VoyageReranker
from .chunking import chunk_text

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
        logger.info("[ZEP STORE] ADDING MESSAGES TO ZEP CLOUD")
        
        try:
            # Ensure thread exists
            thread_id = await self._ensure_thread_exists(user_id)
            
            logger.info(f"[ZEP STORE] Thread ID: {thread_id}")
            logger.info(f"[ZEP STORE] User ID: {user_id}")
            
            # Log full message content
            logger.info("[ZEP STORE] MESSAGE PAYLOAD TO ZEP:")
            logger.info("[ZEP STORE] Message 1 (role=user):")
            for line in user_message.split('\n'):
                logger.info(f"  | {line}")
            logger.info("[ZEP STORE] Message 2 (role=assistant):")
            for line in assistant_response.split('\n'):
                logger.info(f"  | {line}")
            
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
            
            logger.info(f"[ZEP STORE] Sending {len(messages)} messages to Zep API...")
            logger.info(f"[ZEP STORE] User message length: {len(user_message)} chars")
            logger.info(f"[ZEP STORE] Assistant response length: {len(assistant_response)} chars")
            
            response = self.zep.thread.add_messages(
                thread_id=thread_id,
                messages=messages,
                return_context=return_context,
            )
            
            logger.info(f"[ZEP STORE] [OK] Messages added to Zep successfully!")
            logger.info(f"[ZEP STORE] Zep will now extract facts and entities in background")
            
            if return_context and hasattr(response, 'context'):
                logger.info(f"[ZEP STORE] Received updated context block ({len(response.context)} chars)")
                return response.context
            
            logger.info("[ZEP STORE] No context returned (return_context=False)")
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
        logger.info("=" * 70)
        logger.info("[ZEP RETRIEVE] GETTING CONTEXT FROM ZEP CLOUD")
        logger.info("=" * 70)
        
        try:
            thread_id = self._get_thread_id(user_id)
            
            logger.info(f"[ZEP RETRIEVE] User ID: {user_id}")
            logger.info(f"[ZEP RETRIEVE] Thread ID: {thread_id}")
            
            # Check if thread exists first
            if thread_id not in self._created_threads:
                try:
                    logger.info(f"[ZEP RETRIEVE] Checking if thread exists in Zep...")
                    self.zep.thread.get(thread_id=thread_id)
                    self._created_threads.add(thread_id)
                    logger.info(f"[ZEP RETRIEVE] Thread found in Zep")
                except NotFoundError:
                    logger.info(f"[ZEP RETRIEVE] Thread does not exist: {thread_id}")
                    logger.info("[ZEP RETRIEVE] This is a NEW USER - no memory context available")
                    return ""
            
            logger.info(f"[ZEP RETRIEVE] Calling Zep API: get_user_context()...")
            
            user_context = self.zep.thread.get_user_context(thread_id=thread_id)
            
            if user_context and hasattr(user_context, 'context') and user_context.context:
                context = user_context.context
                logger.info(f"[ZEP RETRIEVE] [OK] Context retrieved from Zep!")
                logger.info(f"[ZEP RETRIEVE] Context length: {len(context)} characters")
                logger.info(f"[ZEP RETRIEVE] Context: {context}")
                logger.info("-" * 50)
                logger.info("[ZEP RETRIEVE] FULL CONTEXT DATA FROM ZEP:")
                logger.info("-" * 50)
                for line in context.split('\n'):
                    logger.info(f"  | {line}")
                logger.info("-" * 50)
                return context
            
            logger.info("[ZEP RETRIEVE] No context found (user has no stored memories yet)")
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
        source: str = "document"
    ) -> str:
        """
        Add large document to Zep with automatic chunking.
        
        Handles Zep's 10,000 character limit by splitting documents
        into smaller chunks with overlap for context continuity.
        
        Args:
            user_id: User identifier
            data: The document content (can be any size)
            source: Source/name of the document
        
        Returns:
            Status message with chunk count
        """
        logger.info("=" * 70)
        logger.info("[DOCUMENT] ADDING BUSINESS DATA TO ZEP (WITH CHUNKING)")
        logger.info("=" * 70)
        
        try:
            logger.info(f"[DOCUMENT] User ID: {user_id}")
            logger.info(f"[DOCUMENT] Source: {source}")
            logger.info(f"[DOCUMENT] Total Data Length: {len(data)} characters")
            
            # Ensure user exists by creating a thread first
            await self._ensure_thread_exists(user_id)
            
            # 1. Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            logger.info(f"[DOCUMENT] Generated Document ID: {doc_id}")
            
            # 2. Chunk the data (9000 chars with 50 char overlap)
            chunks = chunk_text(data, chunk_size=9000, overlap=50)
            total_chunks = len(chunks)
            
            logger.info(f"[DOCUMENT] Split into {total_chunks} chunk(s)")
            
            # 3. Add each chunk to Zep graph
            for idx, chunk_content in enumerate(chunks):
                chunk_data = json.dumps({
                    "content": chunk_content,
                    "doc_id": doc_id,
                    "chunk_index": idx + 1,
                    "total_chunks": total_chunks,
                    "source": source
                })
                
                self.zep.graph.add(
                    user_id=user_id,
                    type="json",
                    data=chunk_data
                )
                logger.info(f"[DOCUMENT] [OK] Added chunk {idx + 1}/{total_chunks} ({len(chunk_content)} chars)")
            
            success_msg = f"Document added successfully ({total_chunks} chunks, doc_id: {doc_id})"
            logger.info(f"[DOCUMENT] {success_msg}")
            return success_msg
            
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
