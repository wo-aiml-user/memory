"""
Chat Pydantic Models
Request and response models for the chat API.
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ChatStatus(str, Enum):
    """Status of chat processing."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_id: str
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    user_id: str
    response: str
    status: ChatStatus = ChatStatus.COMPLETED
    error: Optional[str] = None


class MemoryResult(BaseModel):
    """Result from memory retrieval."""
    facts: list[str]
    relevance_scores: Optional[list[float]] = None
