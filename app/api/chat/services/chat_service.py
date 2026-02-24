"""
Chat Service
Async chat service with background processing.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from app.memory.memory_chain import MemoryChain
from app.api.chat.models.chat_model import ChatStatus

logger = logging.getLogger("memory_chat.service")


@dataclass
class ChatResult:
    """Result of a chat operation."""
    user_id: str
    response: str
    status: ChatStatus
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ChatService:
    """
    Async chat service with background task processing.
    All operations are async with try/except and logging.
    """
    
    def __init__(self, memory_chain: MemoryChain):
        """
        Initialize the chat service.
        
        Args:
            memory_chain: Memory chain for chat processing
        """
        self.memory_chain = memory_chain
        self._results: dict[str, ChatResult] = {}
        
        logger.info("ChatService initialized")
    
    async def process_chat(self, user_id: str, message: str) -> ChatResult:
        """
        Process a chat message synchronously.
        
        Args:
            user_id: User identifier
            message: User's message
        
        Returns:
            ChatResult with response and token usage
        """
        try:
            logger.info(f"[process_chat] Starting for user: {user_id}")
            logger.debug(f"[process_chat] Message: {message}")
            
            # Call the memory chain — returns (response_text, usage_dict)
            response, usage = await self.memory_chain.chat(user_id, message)
            
            result = ChatResult(
                user_id=user_id,
                response=response,
                status=ChatStatus.COMPLETED,
                token_usage=usage,
            )
            
            logger.info(f"[process_chat] Completed for user: {user_id}")
            return result
            
        except Exception as e:
            logger.exception(f"[process_chat] Error for user {user_id}: {e}")
            return ChatResult(
                user_id=user_id,
                response="",
                status=ChatStatus.ERROR,
                error=str(e),
            )
    
    async def process_chat_background(
        self, 
        task_id: str, 
        user_id: str, 
        message: str
    ) -> None:
        """
        Process a chat message in the background.
        Stores result for later retrieval.
        
        Args:
            task_id: Unique task identifier
            user_id: User identifier
            message: User's message
        """
        try:
            logger.info(f"[background] Starting task: {task_id}")
            
            # Mark as processing
            self._results[task_id] = ChatResult(
                user_id=user_id,
                response="",
                status=ChatStatus.PROCESSING,
            )
            
            # Process the chat
            result = await self.process_chat(user_id, message)
            
            # Store the result
            self._results[task_id] = result
            
            logger.info(f"[background] Completed task: {task_id}")
            
        except Exception as e:
            logger.exception(f"[background] Error in task {task_id}: {e}")
            self._results[task_id] = ChatResult(
                user_id=user_id,
                response="",
                status=ChatStatus.ERROR,
                error=str(e),
            )
    
    def get_result(self, task_id: str) -> Optional[ChatResult]:
        """
        Get the result of a background task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            ChatResult if available, None otherwise
        """
        return self._results.get(task_id)
    
    def clear_result(self, task_id: str) -> None:
        """
        Clear a stored result.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self._results:
            del self._results[task_id]
            logger.debug(f"[clear_result] Cleared: {task_id}")
