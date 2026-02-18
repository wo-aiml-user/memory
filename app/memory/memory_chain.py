"""
Memory Chain
Main orchestration logic for memory-enhanced chat using Gemini SDK and Manual Tool Execution.
Follows the official "Function Calling" steps:
1. Define declarations (Tools).
2. Call model with declarations.
3. Execute function code.
4. Send result back to model.
"""

import logging
import os
import asyncio
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types

# Import tools and init
from .memory_tools import init_tools, retrieve_memory_tool, save_chat_message, check_and_store_memory

logger = logging.getLogger("memory_chat.chain")

class MemoryChain:
    """
    Agentic Memory Chain using Gemini SDK with Manual Function Calling Loop.
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
        
        db_name = os.environ.get("MONGODB_DB_NAME", "memory")
        collection_name = os.environ.get("MONGODB_COLLECTION_NAME", "user-memory")

        # Initialize the global tools
        init_tools(uri, voyage_key, self.client, db_name, collection_name, model_name=model)

        self.tools_list = [retrieve_memory_tool]
        
        # Create a mapping for execution (Step 3)
        self.tools_map = {
            "retrieve_memory_tool": retrieve_memory_tool
        }

        logger.info(f"MemoryChain (Gemini SDK Manual Loop) initialized. DB: {db_name}.{collection_name}")

    async def chat(self, user_id: str, message: str, chat_history: List[Dict] = None) -> str:
        """
        Process a chat message using the Manual Function Calling Loop.
        
        Args:
            user_id: User identifier.
            message: The user's message.
            chat_history: List of past messages.
        
        Returns:
            Assistant response string.
        """
        logger.info(f"Processing message for user {user_id}")
        
        # --- 1. Auto-Save User Message ---
        await save_chat_message(user_id, "user", message)
        

        store_result = await check_and_store_memory(user_id)
        if "Triggering memory storage" in str(store_result):
             logger.info(f"Memory update triggered: {store_result}")

        # System Instruction
        system_instruction = (
            "You are a helpful assistant with a distinct personality. "
            "You have access to long-term memory tools. "
            "Whenever relevant, use 'retrieve_memory_tool' to recall past user details. "
            "Do NOT try to store memories yourself; this is done automatically. "
            f"Current User ID: {user_id}"
        )

        # Config (Step 2: Call model with function declarations)
        config = types.GenerateContentConfig(
            tools=self.tools_list,
            system_instruction=system_instruction,
            temperature=0.7,
            # automatic_function_calling=dict(disable=False) # DISABLED for manual loop
        )
        
        # Build Initial Context
        contents = []
        if chat_history:
             for msg in chat_history:
                 role = msg.get("role", "user")
                 text = msg.get("content", "")
                 if role == "assistant": role = "model"
                 contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        
        try:
            # === Manual Function Calling Loop ===
            
            # Initial Call
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Use a loop to handle potential chained calls (though usually 1 turn is enough)
            max_turns = 5
            final_response_text = ""
            
            for _ in range(max_turns):
                
                # Check for function calls (Step 3: Execute function code)
                function_calls = response.function_calls
                
                if function_calls:
                    if self.verbose:
                        logger.info(f"Function Call identified: {function_calls}")
                    
                    # Store models response (the call) in history for the next turn
                    contents.append(response.candidates[0].content)
                    
                    # Execute all calls in this turn
                    function_responses_parts = []
                    
                    for call in function_calls:
                        func_name = call.name
                        func_args = call.args
                        
                        tool_func = self.tools_map.get(func_name)
                        
                        if tool_func:
                            try:
                                print(f"\n[TOOL CALL] Executing: {func_name}")
                                print(f"[TOOL ARGS] {func_args}")
                                logger.info(f"Executing {func_name} with args: {func_args}")
                                    
                                # Execute (Async)
                                result = await tool_func(**func_args)
                                
                                print(f"[TOOL RESULT] {result}\n")
                                logger.info(f"Tool Result: {result}")
                                
                                # Create Function Response Part
                                function_responses_parts.append(
                                    types.Part.from_function_response(
                                        name=func_name,
                                        response={"result": result}
                                    )
                                )
                                
                            except Exception as e:
                                logger.error(f"Error executing tool {func_name}: {e}")
                                function_responses_parts.append(
                                    types.Part.from_function_response(
                                        name=func_name,
                                        response={"error": str(e)}
                                    )
                                )
                        else:
                             # Tool not found
                             function_responses_parts.append(
                                types.Part.from_function_response(
                                    name=func_name,
                                    response={"error": "Tool not found"}
                                )
                            )

                    # Step 4: Create user friendly response with function result and call the model again
                    # Append function responses to contents
                    contents.append(types.Content(role="tool", parts=function_responses_parts))
                    
                    # Call model again
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    )
                    
                else:
                    # No function calls, just text
                    final_response_text = response.text
                    break
            
            if not final_response_text:
                final_response_text = response.text

            # --- 3. Auto-Save Assistant Response ---
            await save_chat_message(user_id, "model", final_response_text)
            
            return final_response_text
            
        except Exception as e:
            logger.error(f"Error during Gemini SDK execution: {e}")
            return f"Error: {str(e)}"
