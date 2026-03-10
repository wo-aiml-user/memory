"""
Memory Tools
Agent-facing tools for file-based memory:
- write_memory_tool
- get_memory_tool
"""

import logging
import os
from typing import Optional

from .embedding import VoyageEmbedder
from .file_memory_store import FileMemoryStore

logger = logging.getLogger("memory_chat.tools")

_memory_store: Optional[FileMemoryStore] = None


def init_tools(
    voyage_api_key: Optional[str],
    workspace_dir: str = "workspace",
    assistant_id: str = "main",
) -> None:
    """Initialize file-memory tool backend."""
    global _memory_store
    embedder = None
    logger.info(
        "[TOOL_INIT] init_tools called workspace_dir=%s assistant_id=%s voyage_key_present=%s",
        workspace_dir,
        assistant_id,
        bool(voyage_api_key),
    )

    if voyage_api_key:
        try:
            embedder = VoyageEmbedder(api_key=voyage_api_key)
            logger.info("Memory tools using semantic retrieval with Voyage embeddings")
        except Exception as e:
            logger.warning("Failed to initialize Voyage embedder. Semantic retrieval will be unavailable: %s", e)
    else:
        logger.warning("VOYAGE_API_KEY missing. Semantic retrieval will be unavailable.")

    _memory_store = FileMemoryStore(
        base_dir=workspace_dir,
        assistant_id=assistant_id,
        embedder=embedder,
    )
    logger.info("[TOOL_INIT] file memory tools initialized workspace_dir=%s", workspace_dir)


async def write_memory_tool(
    content: str,
    user_id: str,
    memory_type: str = "daily",
    topic: str = "general",
) -> str:
    """
    Write new memory into the file-based memory system.

    Args:
        content: The memory content to save.
        user_id: Current user id.
        memory_type: daily | projects | curated
        topic: Topic label for organization.
    """
    if _memory_store is None:
        return "Error: Memory tools not initialized."
    logger.info(
        "[TOOL_CALL] write_memory_tool user_id=%s memory_type=%s topic=%s content=%s",
        user_id,
        memory_type,
        topic,
        content,
    )
    try:
        target = await _memory_store.write_memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            topic=topic,
        )
        result = f"Memory saved in {target}"
        logger.info("[TOOL_RESULT] write_memory_tool result=%s", result)
        return result
    except Exception as e:
        logger.error("write_memory_tool failed: %s", e)
        return f"Error writing memory: {e}"


async def get_memory_tool(
    query: str,
    user_id: str,
    scope: str = "all",
    top_k: int = 5,
) -> str:
    """
    Retrieve relevant memory using semantic retrieval.

    Args:
        query: User query for memory lookup.
        user_id: Current user id.
        scope: startup | all
        top_k: Number of chunks to return.
    """
    logger.info(
        "[TOOL_CALL] get_memory_tool user_id=%s query=%s scope=%s top_k=%s",
        user_id,
        query,
        scope,
        top_k,
    )
    del user_id  # Reserved for future user-specific partitioning.

    if _memory_store is None:
        return "Error: Memory tools not initialized."

    try:
        hits = await _memory_store.retrieve(query=query, scope=scope, top_k=top_k)
        if not hits:
            logger.info("[TOOL_RESULT] get_memory_tool no relevant memory found")
            return "No relevant memory found."

        lines = ["Relevant memory:"]
        for hit in hits:
            preview = hit.text.strip().replace("\n", " ")
            if len(preview) > 400:
                preview = preview[:400] + "..."
            lines.append(f"- [{hit.score:.3f}] {hit.file_path}: {preview}")
        result = "\n".join(lines)
        logger.info("[TOOL_RESULT] get_memory_tool result=%s", result)
        return result
    except Exception as e:
        logger.error("get_memory_tool failed: %s", e)
        return f"Error getting memory: {e}"


async def append_chat_log(user_id: str, role: str, content: str) -> None:
    """Internal helper for append-only daily logs."""
    if _memory_store is None:
        return
    logger.info(
        "[MEMORY_EVENT] append_chat_log requested user_id=%s role=%s content=%s",
        user_id,
        role,
        content,
    )
    try:
        await _memory_store.append_chat_log(user_id=user_id, role=role, content=content)
        logger.info("[MEMORY_EVENT] append_chat_log stored user_id=%s role=%s", user_id, role)
    except Exception as e:
        logger.warning("append_chat_log failed: %s", e)


async def get_startup_context(user_id: str, query: str, top_k: int = 4) -> str:
    """Retrieve startup context from MEMORY.md + today/yesterday logs."""
    logger.info(
        "[MEMORY_EVENT] get_startup_context user_id=%s query=%s top_k=%s",
        user_id,
        query,
        top_k,
    )
    result = await get_memory_tool(
        query=query,
        user_id=user_id,
        scope="startup",
        top_k=top_k,
    )
    logger.info("[MEMORY_EVENT] startup context result=%s", result)
    return result


def get_workspace_dir() -> str:
    """Expose workspace dir default for external diagnostics."""
    return os.environ.get("MEMORY_WORKSPACE_DIR", "workspace")


GET_MEMORY_TOOL_SCHEMA = {
    "name": "get_memory_tool",
    "description": "Retrieve relevant memory using semantic retrieval.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User query for memory lookup."},
            "user_id": {"type": "string", "description": "Current user id."},
            "scope": {"type": "string", "description": "startup | all"},
            "top_k": {"type": "integer", "description": "Number of chunks to return."}
        },
        "required": ["query", "user_id"]
    }
}

WRITE_MEMORY_TOOL_SCHEMA = {
    "name": "write_memory_tool",
    "description": "Write new memory into the file-based memory system.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to save."},
            "user_id": {"type": "string", "description": "Current user id."},
            "memory_type": {"type": "string", "description": "daily | projects | curated"},
            "topic": {"type": "string", "description": "Topic label for organization."}
        },
        "required": ["content", "user_id"]
    }
}
