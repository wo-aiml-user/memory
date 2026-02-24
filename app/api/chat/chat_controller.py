"""
Chat Controller
FastAPI router for chat endpoints.
"""

import logging
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.api.chat.models.chat_model import ChatRequest, ChatResponse, ChatStatus, TokenUsage
from app.api.chat.services.chat_service import ChatService

logger = logging.getLogger("memory_chat.controller")

# Router without prefix - prefix is added in route/__init__.py
router = APIRouter()

# Service instance (will be set by main.py)
_chat_service: ChatService = None


def set_chat_service(service: ChatService) -> None:
    """Set the chat service instance."""
    global _chat_service
    _chat_service = service
    logger.info("Chat service set in controller")


def get_chat_service() -> ChatService:
    """Get the chat service instance."""
    if _chat_service is None:
        raise RuntimeError("Chat service not initialized")
    return _chat_service


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Process a chat message with memory.
    
    Args:
        payload: Chat request with user_id and message
        background_tasks: FastAPI background tasks
    
    Returns:
        ChatResponse with the assistant's response and token usage
    """
    try:
        logger.info(f"[/chat] Received request from user: {payload.user_id}")
        logger.debug(f"[/chat] Message: {payload.message}")
        
        service = get_chat_service()
        
        # Process the chat
        result = await service.process_chat(payload.user_id, payload.message)
        
        if result.status == ChatStatus.ERROR:
            logger.error(f"[/chat] Error: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        
        logger.info(f"[/chat] Completed for user: {payload.user_id}")
        
        # Build token_usage model if available
        token_usage = None
        if result.token_usage:
            token_usage = TokenUsage(
                prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                completion_tokens=result.token_usage.get("completion_tokens", 0),
                total_tokens=result.token_usage.get("total_tokens", 0),
            )
        
        return ChatResponse(
            user_id=result.user_id,
            response=result.response,
            token_usage=token_usage,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[/chat] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
