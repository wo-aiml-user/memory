"""
Response Formatter
Utilities for formatting LLM responses and metadata.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass
import re

logger = logging.getLogger("memory_chat.formatter")


@dataclass
class FormattedResponse:
    """Formatted response with metadata."""
    content: str
    memory_retrieved: bool = False
    memory_stored: bool = False
    facts_count: int = 0


def format_llm_response(
    response: str,
    memory_context: Optional[str] = None,
    remove_thinking: bool = True
) -> FormattedResponse:
    """
    Format an LLM response for display.
    
    Args:
        response: Raw LLM response
        memory_context: Optional memory context that was used
        remove_thinking: Whether to remove thinking/reasoning tags
    
    Returns:
        FormattedResponse with cleaned content
    """
    try:
        content = response
        
        # Remove thinking tags if present (e.g., <think>...</think>)
        if remove_thinking:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        
        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        
        # Detect if memory was used
        memory_retrieved = memory_context is not None and "No relevant memories" not in memory_context
        facts_count = memory_context.count("- ") if memory_context else 0
        
        logger.debug(f"Formatted response: {len(content)} chars, {facts_count} facts")
        
        return FormattedResponse(
            content=content,
            memory_retrieved=memory_retrieved,
            facts_count=facts_count,
        )
        
    except Exception as e:
        logger.exception(f"Error formatting response: {e}")
        return FormattedResponse(content=response)


def format_conversation_for_storage(user_message: str, assistant_response: str) -> str:
    """
    Format a conversation exchange for storage in memory.
    
    Args:
        user_message: User's message
        assistant_response: Assistant's response
    
    Returns:
        Formatted conversation string
    """
    return f"User: {user_message}\nAssistant: {assistant_response}"


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
