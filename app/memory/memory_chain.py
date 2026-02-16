"""
Memory Chain
Main orchestration logic for memory-enhanced chat using Gemini SDK and Tools.
"""

import logging
import os
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types

# Import new tools and init
from .memory_tools import init_tools, store_memory_tool, retrieve_memory_tool

logger = logging.getLogger("memory_chat.chain")

class MemoryChain:
    """
    Agentic Memory Chain using Gemini SDK.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        mongo_uri: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.model_name = model
        
        # 1. Setup Gemini Client
        google_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not google_api_key:
            logger.warning("GEMINI_API_KEY missing!")
            
        self.client = genai.Client(api_key=google_api_key)
        
        # 2. Setup Tools
        voyage_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
        if not voyage_key:
            logger.warning("Voyage API Key missing! Embeddings will fail.")
            
        # Get Mongo config from Env
        uri = mongo_uri or os.environ.get("MONGODB_URI")
        if not uri:
             uri = "mongodb://localhost:27017"
        
        db_name = os.environ.get("MONGODB_DB_NAME", "memory")
        collection_name = os.environ.get("MONGODB_COLLECTION_NAME", "user-memory")

        # Initialize the global tools
        # We also pass the same client/model to the tools for internal LLM usage (extraction)
        init_tools(uri, voyage_key, self.client, db_name, collection_name, model_name=model)
        
        # Tool Declarations for Gemini
        # We pass the functions directly. The SDK inspects them.
        self.tools = [store_memory_tool, retrieve_memory_tool]

        logger.info(f"MemoryChain (Gemini SDK) initialized. DB: {db_name}.{collection_name}")

    async def chat(self, user_id: str, message: str, chat_history: List[Dict] = None) -> str:
        """
        Process a chat message using Gemini with Tools.
        
        Args:
            user_id: User identifier.
            message: The user's message.
            chat_history: List of past messages (optional). 
                          Expected format: [{"role": "user", "content": "..."}, ...]
                          or the types.Content format.
        
        Returns:
            Assistant response string.
        """
        logger.info(f"Processing message for user {user_id}")
        
        # Prepare context/instructions
        system_instruction = (
            "You are a helpful assistant with specific long-term memory capabilities. "
            "You MUST use 'retrieve_memory_tool' to look up information about the user, past conversations, or facts when relevant. "
            "You MUST use 'store_memory_tool' to save important facts, preferences, or events from the conversation into long-term memory. "
            "Do not rely only on current context; actively use your memory tools. "
            f"Current User ID: {user_id}"
        )

        # Config with Tools
        # Note: The 'google-genai' SDK allows passing python functions directly in `tools`.
        config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=system_instruction,
            temperature=0.7,
            automatic_function_calling=dict(disable=False, maximum_remote_calls=5) # Enable auto tool execution
        )
        
        # Build contents
        contents = []
        if chat_history:
             # Convert simple dict history to types.Content if needed, or pass as is if SDK supports it.
             # Assuming chat_history is list of Content objects or compatible dicts.
             for msg in chat_history:
                 role = msg.get("role", "user")
                 text = msg.get("content", "")
                 if role == "assistant": role = "model"
                 contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

        try:
            # We use a chat session for convenience if we want to maintain state, 
            # but here we are stateless per request (using passed history).
            # generate_content is easier for stateless.
            
            # NOTE: 'google-genai' SDK's automatic_function_calling handles the loop!
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error during Gemini SDK execution: {e}")
            return f"Error: {str(e)}"
