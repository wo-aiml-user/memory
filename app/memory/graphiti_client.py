"""
Graphiti Client
Manages connection to Neo4j via Graphiti for graph-based memory.
"""

import logging
import asyncio
from abc import ABC
from datetime import datetime
from typing import Optional, Iterable

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.client import EmbedderClient

from .embedding import VoyageEmbedder
from .reranker import VoyageReranker

logger = logging.getLogger("memory_chat.graphiti")


class VoyageGraphitiEmbedder(EmbedderClient):
    """
    Custom embedder adapter for Graphiti that uses Voyage AI.
    Inherits from EmbedderClient to satisfy Graphiti's type requirements.
    """
    
    def __init__(self, voyage_embedder: VoyageEmbedder):
        self.embedder = voyage_embedder
    
    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single text input."""
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and all(isinstance(t, str) for t in input_data):
            texts = input_data
        else:
            texts = [str(input_data)]
        
        embeddings = await self.embedder.embed_documents(texts)
        return embeddings[0] if embeddings else []
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""
        return await self.embedder.embed_documents(input_data_list)


class GraphitiMemoryClient:
    """
    Graphiti client for graph-based long-term memory.
    Uses Voyage AI for embeddings and reranking, DeepSeek for LLM.
    
    Key features:
    - Proper episode_body formatting with User/Assistant roles
    - Async ingestion to avoid blocking response flow
    - group_id for multi-tenant user separation
    - Proper timestamp handling
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        voyage_api_key: str,
        deepseek_api_key: str,
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_model: str = "deepseek-chat",
        voyage_embed_model: str = "voyage-3-large",
        voyage_rerank_model: str = "rerank-2.5",
    ):
        """
        Initialize the Graphiti memory client.
        """
        logger.info("Initializing GraphitiMemoryClient")
        
        # Initialize Voyage AI components
        self.voyage_embedder = VoyageEmbedder(voyage_api_key, voyage_embed_model)
        self.voyage_reranker = VoyageReranker(voyage_api_key, voyage_rerank_model)
        
        # Create Graphiti-compatible embedder
        graphiti_embedder = VoyageGraphitiEmbedder(self.voyage_embedder)
        
        # Remove /v1 suffix if present - OpenAI SDK adds it automatically
        if deepseek_base_url.endswith("/v1"):
            deepseek_base_url = deepseek_base_url[:-3]
        deepseek_base_url = deepseek_base_url.rstrip("/")
        
        # Create DeepSeek LLM client for Graphiti
        logger.info(f"Configuring DeepSeek LLM client with model: {deepseek_model}, base_url: {deepseek_base_url}")
        llm_config = LLMConfig(
            api_key=deepseek_api_key,
            model=deepseek_model,
            base_url=deepseek_base_url,
        )
        deepseek_llm_client = OpenAIGenericClient(config=llm_config)
        
        # Create cross-encoder/reranker
        reranker_config = LLMConfig(
            api_key=deepseek_api_key,
            model=deepseek_model,
            base_url=deepseek_base_url,
        )
        deepseek_cross_encoder = OpenAIRerankerClient(config=reranker_config)
        
        # Initialize Graphiti with Neo4j connection
        self.graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=deepseek_llm_client,
            embedder=graphiti_embedder,
            cross_encoder=deepseek_cross_encoder,
        )
        
        self._initialized = False
        # Track background ingestion tasks
        self._pending_tasks: set[asyncio.Task] = set()
        
        logger.info("GraphitiMemoryClient initialized")
    
    async def initialize(self) -> None:
        """Initialize Graphiti indices and constraints."""
        try:
            if not self._initialized:
                logger.info("Building Graphiti indices and constraints")
                await self.graphiti.build_indices_and_constraints()
                self._initialized = True
                logger.info("Graphiti indices built successfully")
        except Exception as e:
            logger.exception(f"Error initializing Graphiti: {e}")
            raise
    
    @staticmethod
    def format_episode_body(
        user_message: str,
        assistant_response: str,
        user_name: str = "User",
        assistant_name: str = "Assistant"
    ) -> str:
        """
        Format conversation into episode_body string.
        
        Graphiti expects consistent role names for proper entity extraction.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            user_name: Name for user role (default: "User")
            assistant_name: Name for assistant role (default: "Assistant")
        
        Returns:
            Formatted episode body string
        """
        return f"{user_name}: {user_message}\n{assistant_name}: {assistant_response}"
    
    async def store_memory(
        self,
        user_id: str,
        conversation: str,
        source_description: str = "Chat conversation",
        reference_time: Optional[datetime] = None,
        async_ingestion: bool = True
    ) -> str:
        """
        Store a conversation as memory in the graph.
        
        Uses add_episode() which automatically:
        - Creates an episodic node with raw conversation
        - Extracts entities (people, places, concepts) using LLM
        - Deduplicates entities against existing nodes
        - Extracts relationships between entities
        - Links episode to extracted entities
        
        Args:
            user_id: User identifier (used as group_id for multi-tenant)
            conversation: Formatted conversation ("User: ...\\nAssistant: ...")
            source_description: Description of the source
            reference_time: Actual conversation time (None = now)
            async_ingestion: If True, don't block response flow
        
        Returns:
            Status message
        """
        episode_time = reference_time or datetime.now()
        episode_name = f"conversation_{user_id}_{episode_time.timestamp()}"
        
        logger.info(f"[store_memory] Storing memory for user: {user_id}")
        logger.debug(f"[store_memory] Episode: {episode_name}")
        logger.debug(f"[store_memory] Content: {conversation}")
        
        async def _ingest():
            """Background ingestion task."""
            try:
                await self.graphiti.add_episode(
                    name=episode_name,
                    episode_body=conversation,
                    source_description=f"{source_description} with {user_id}",
                    reference_time=episode_time,
                    source=EpisodeType.message,
                    group_id=user_id,  # Essential for multi-tenant separation
                )
                logger.info(f"[store_memory] Successfully stored episode: {episode_name}")
            except Exception as e:
                logger.exception(f"[store_memory] Error storing episode {episode_name}: {e}")
        
        if async_ingestion:
            # Non-blocking: use asyncio.create_task() to avoid blocking response
            task = asyncio.create_task(_ingest())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            logger.info(f"[store_memory] Queued async ingestion for episode: {episode_name}")
            return "Memory queued for storage."
        else:
            # Blocking: wait for ingestion to complete
            try:
                await _ingest()
                return "Memory stored successfully."
            except Exception as e:
                return f"Error storing memory: {str(e)}"
    
    async def store_multi_turn_memory(
        self,
        user_id: str,
        turns: list[tuple[str, str]],
        source_description: str = "Multi-turn conversation",
        reference_time: Optional[datetime] = None,
        async_ingestion: bool = True
    ) -> str:
        """
        Store multiple conversation turns as a single episode.
        
        Use this for related turns that should be stored together.
        
        Args:
            user_id: User identifier
            turns: List of (user_message, assistant_response) tuples
            source_description: Description of the source
            reference_time: Actual conversation time
            async_ingestion: If True, don't block response flow
        
        Returns:
            Status message
        """
        # Format all turns into single episode body
        episode_body_parts = []
        for user_msg, assistant_resp in turns:
            episode_body_parts.append(f"User: {user_msg}")
            episode_body_parts.append(f"Assistant: {assistant_resp}")
        
        conversation = "\n".join(episode_body_parts)
        
        return await self.store_memory(
            user_id=user_id,
            conversation=conversation,
            source_description=source_description,
            reference_time=reference_time,
            async_ingestion=async_ingestion
        )
    
    async def store_document(
        self,
        user_id: str,
        file_id: str,
        file_name: str,
        content: str,
        source_description: Optional[str] = None,
        reference_time: Optional[datetime] = None,
        async_ingestion: bool = True
    ) -> str:
        """
        Store uploaded document content as a memory episode.
        
        Uses EpisodeType.text for unstructured text from documents.
        
        Graphiti automatically:
        - Creates an episodic node with the document content
        - Extracts entities (people, places, concepts) using LLM
        - Deduplicates entities against existing nodes
        - Extracts relationships between entities
        
        Args:
            user_id: User identifier (group_id for multi-tenant)
            file_id: Unique file identifier
            file_name: Name of the uploaded file
            content: Extracted text content from document
            source_description: Description (default: "Document: {file_name}")
            reference_time: Upload timestamp (None = now)
            async_ingestion: If True, don't block response
        
        Returns:
            Status message
        """
        episode_time = reference_time or datetime.now()
        episode_name = f"document_{user_id}_{file_id}_{episode_time.timestamp()}"
        source_desc = source_description or f"Document: {file_name}"
        
        logger.info(f"[store_document] Storing for user: {user_id}, file: {file_name}")
        logger.debug(f"[store_document] Content length: {len(content)} chars")
        
        async def _ingest():
            try:
                await self.graphiti.add_episode(
                    name=episode_name,
                    episode_body=content,
                    source_description=source_desc,
                    reference_time=episode_time,
                    source=EpisodeType.text,  # For unstructured documents
                    group_id=user_id,
                )
                logger.info(f"[store_document] Stored: {episode_name}")
            except Exception as e:
                logger.exception(f"[store_document] Error: {e}")
        
        if async_ingestion:
            task = asyncio.create_task(_ingest())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return "Document queued for storage."
        else:
            try:
                await _ingest()
                return "Document stored successfully."
            except Exception as e:
                return f"Error storing document: {str(e)}"
    
    async def store_document_chunks(
        self,
        user_id: str,
        file_id: str,
        file_name: str,
        chunks: list[str],
        source_description: Optional[str] = None,
        reference_time: Optional[datetime] = None,
        async_ingestion: bool = True
    ) -> str:
        """
        Store large documents as multiple chunk episodes.
        
        Args:
            user_id: User identifier
            file_id: Unique file identifier
            file_name: Name of the uploaded file
            chunks: List of text chunks from the document
            source_description: Description of the source
            reference_time: Upload timestamp
            async_ingestion: If True, don't block response
        
        Returns:
            Status message
        """
        logger.info(f"[store_chunks] Storing {len(chunks)} chunks for: {file_name}")
        
        episode_time = reference_time or datetime.now()
        source_desc = source_description or f"Document: {file_name}"
        
        async def _ingest_all():
            for i, chunk in enumerate(chunks):
                episode_name = f"doc_{user_id}_{file_id}_c{i}_{episode_time.timestamp()}"
                try:
                    await self.graphiti.add_episode(
                        name=episode_name,
                        episode_body=chunk,
                        source_description=f"{source_desc} (chunk {i+1}/{len(chunks)})",
                        reference_time=episode_time,
                        source=EpisodeType.text,
                        group_id=user_id,
                    )
                except Exception as e:
                    logger.exception(f"[store_chunks] Chunk {i+1} error: {e}")
            logger.info(f"[store_chunks] Completed {len(chunks)} chunks")
        
        if async_ingestion:
            task = asyncio.create_task(_ingest_all())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return f"Document ({len(chunks)} chunks) queued."
        else:
            try:
                await _ingest_all()
                return f"Document ({len(chunks)} chunks) stored."
            except Exception as e:
                return f"Error: {str(e)}"
    
    async def retrieve_memory(
        self,
        user_id: str,
        query: str,
        num_results: int = 5,
        use_reranking: bool = True
    ) -> str:
        """
        Retrieve relevant memories from the graph.
        
        Args:
            user_id: User identifier for filtering memories
            query: Search query
            num_results: Number of results to return
            use_reranking: Whether to apply Voyage AI reranking
        
        Returns:
            Formatted string of relevant memories
        """
        try:
            logger.info(f"[retrieve_memory] Searching for user: {user_id}, query: {query}")
            
            # Search Graphiti with group_id filter
            results = await self.graphiti.search(
                query=query,
                group_ids=[user_id],
                num_results=num_results * 2 if use_reranking else num_results,
            )
            
            if not results:
                logger.info("[retrieve_memory] No memories found")
                return "No relevant memories found."
            
            # Extract facts from results
            facts = [edge.fact for edge in results]
            
            # Apply Voyage AI reranking if enabled
            if use_reranking and len(facts) > 1:
                logger.debug(f"[retrieve_memory] Reranking {len(facts)} results")
                reranked = await self.voyage_reranker.rerank(query, facts, top_k=num_results)
                facts = [r.document for r in reranked]
            
            formatted = "Relevant memories:\n" + "\n".join(f"- {f}" for f in facts[:num_results])
            logger.info(f"[retrieve_memory] Found {len(facts[:num_results])} relevant memories")
            return formatted
            
        except Exception as e:
            logger.exception(f"[retrieve_memory] Error retrieving memories: {e}")
            return f"Error retrieving memories: {str(e)}"
    
    async def wait_for_pending_tasks(self, timeout: float = 30.0) -> None:
        """
        Wait for all pending async ingestion tasks to complete.
        
        Call this during shutdown to ensure all memories are stored.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._pending_tasks:
            logger.info(f"Waiting for {len(self._pending_tasks)} pending ingestion tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pending_tasks, return_exceptions=True),
                    timeout=timeout
                )
                logger.info("All pending ingestion tasks completed")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for pending tasks after {timeout}s")
    
    async def close(self) -> None:
        """Close the Graphiti connection."""
        try:
            # Wait for pending ingestion tasks
            await self.wait_for_pending_tasks()
            
            await self.graphiti.close()
            logger.info("Graphiti connection closed")
        except Exception as e:
            logger.exception(f"Error closing Graphiti connection: {e}")
