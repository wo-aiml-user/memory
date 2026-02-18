"""
Memory Tools
Tools for updating and retrieving memories.
Refactored to be used with Gemini SDK directly (no LangChain).
"""

import logging
from typing import List, Optional, Dict, Any

# We still need the clients
from .mongo_client import MongoMemoryClient
from .embedding import VoyageEmbedder
from google import genai
from google.genai import types

logger = logging.getLogger("memory_chat.tools")

# Global instances
_mongo_client: Optional[MongoMemoryClient] = None
_embedder: Optional[VoyageEmbedder] = None
_genai_client: Optional[genai.Client] = None
_model_name: str = "gemini-2.5-flash"

def init_tools(mongo_uri: str, voyage_api_key: str, genai_client: genai.Client, db_name: str = "memory", collection_name: str = "user-memory", model_name: str = "gemini-2.5-flash"):
    """Initialize the tools with necessary clients."""
    global _mongo_client, _embedder, _genai_client, _model_name
    _mongo_client = MongoMemoryClient(uri=mongo_uri, db_name=db_name, collection_name=collection_name)
    _embedder = VoyageEmbedder(api_key=voyage_api_key)
    _genai_client = genai_client
    _model_name = model_name
    logger.info(f"Memory tools initialized for {db_name}.{collection_name}")

async def store_memory_tool(conversation_history: str, user_id: str) -> str:
    """
    Stores relevant information from the conversation history into long-term memory.
    Call this when the user mentions new preferences, facts, or important details.
    
    Args:
        conversation_history: The recent conversation text or specific facts to store.
        user_id: The ID of the user.
    """
    if not _mongo_client or not _embedder or not _genai_client:
        return "Error: Memory tools not initialized."

    logger.info(f"Storing memory for user {user_id}")

    # 1. format/extract using LLM (SDK direct)
    extraction_prompt = f"""
    Analyze the following conversation or text and extract key distinct facts, preferences, or events that should be stored in long-term memory.
    Return them as a single concise paragraph or list of statements.
    
    Text:
    {conversation_history}
    """
    
    try:
        response = _genai_client.models.generate_content(
            model=_model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"System: You are a memory manager.\n{extraction_prompt}")]
                )
            ]
        )
        formatted_content = response.text
        if not formatted_content:
            formatted_content = conversation_history # Fallback
            
        # 2. Embed
        embedding = await _embedder.embed_query(formatted_content)
        
        # 3. Store
        doc_id = _mongo_client.add_memory(
            content=formatted_content,
            embedding=embedding,
            user_id=user_id
        )
        
        print(f"[MEMORY STORE] Storing content: {formatted_content}")
        print(f"[MEMORY ID] {doc_id}")
        return f"Memory stored successfully (ID: {doc_id}). Content: {formatted_content}"

    except Exception as e:
        logger.error(f"Error in store_memory_tool: {e}")
        return f"Error storing memory: {str(e)}"

async def retrieve_memory_tool(query: str, user_id: str) -> str:
    """
    Retrieves relevant past memories based on a search query.
    Call this when you need to recall user preferences, past events, or context.
    
    Args:
        query: The search query (e.g., "what is my favorite color?", "project details").
        user_id: The ID of the user.
    """
    if not _mongo_client or not _embedder:
        return "Error: Memory tools not initialized."

    logger.info(f"Retrieving memory for user {user_id} query='{query}'")

    try:
        # 1. Embed Query
        query_embedding = await _embedder.embed_query(query)
        
        # 2. Search
        results = _mongo_client.search_memories(query_embedding, user_id, limit=5)
        
        if not results:
            print("[MEMORY RETRIEVE] No results found.")
            return "No relevant memories found."
            
        # 3. Format Output
        context_str = "Found relevant memories:\n"
        print(f"[MEMORY RETRIEVE] Query: {query}")
        for doc in results:
            content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            print(f"[MEMORY HIT] (Sim: {doc.get('similarity', 0):.2f}) {content_preview}")
            context_str += f"- {doc['content']} (Similarity: {doc.get('similarity', 0):.2f})\n"
            
        return context_str
        
    except Exception as e:
        logger.error(f"Error in retrieve_memory_tool: {e}")
        return f"Error retrieving memory: {str(e)}"

async def save_chat_message(user_id: str, role: str, content: str) -> str:
    """Save a chat message to history."""
    if _mongo_client:
        return _mongo_client.add_chat_history(user_id, role, content)
    return ""

async def get_chat_history(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get chat history."""
    if _mongo_client:
        return _mongo_client.get_chat_history(user_id, limit)
    return []

async def check_and_store_memory(user_id: str) -> str:
    """
    Check if we have enough messages to trigger memory storage.
    If yes, summarize and store.
    """
    if not _mongo_client or not _embedder or not _genai_client:
        return "Error: Tools not initialized."

    # 1. Fetch recent history (e.g., last 10 messages)
    history = _mongo_client.get_chat_history(user_id, limit=10)
    
    total_count = _mongo_client.chat_collection.count_documents({"user_id": user_id})
    
    if total_count > 0 and total_count % 10 == 0:
        logger.info(f"[AUTO-MEM] Triggering memory storage for user {user_id} (Count: {total_count})")
        
        # Convert history to string
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        # Reuse store logic
        return await store_memory_tool(history_text, user_id)
        
    return "No storage triggered."
