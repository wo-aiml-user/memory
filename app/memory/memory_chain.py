"""
Memory Chain
Gemini SDK manual function-calling loop with file-based memory architecture.
"""

import logging
import os
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from .memory_tools import (
    GET_MEMORY_TOOL_SCHEMA,
    WRITE_MEMORY_TOOL_SCHEMA,
    append_chat_log,
    get_memory_tool,
    init_tools,
    write_memory_tool,
)
from .prompt import system_instruction

logger = logging.getLogger("memory_chat.chain")


class MemoryChain:
    """
    Memory-aware chat chain with file-based persistent memory.

    Startup memory retrieval:
    - workspace/MEMORY.md
    - workspace/memory/<today>.md
    - workspace/memory/<yesterday>.md
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        voyage_api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.model_name = model
        self._session_bootstrapped_users: set[str] = set()

        google_api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not google_api_key:
            logger.warning("GEMINI_API_KEY missing")
        self.client = genai.Client(api_key=google_api_key)

        voyage_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
        workspace_dir = os.environ.get("MEMORY_WORKSPACE_DIR", "workspace")
        assistant_id = os.environ.get("MEMORY_ASSISTANT_ID", "main")

        # Init file-based memory backend
        init_tools(voyage_key, workspace_dir=workspace_dir, assistant_id=assistant_id)

        self.tools_list = [types.Tool(function_declarations=[GET_MEMORY_TOOL_SCHEMA, WRITE_MEMORY_TOOL_SCHEMA])]
        self.tools_map = {
            "get_memory_tool": get_memory_tool,
            "write_memory_tool": write_memory_tool,
        }

        logger.info("MemoryChain initialized with file memory workspace: %s", workspace_dir)

    async def chat(self, user_id: str, message: str, chat_history: List[Dict] = None) -> str:
        """
        Process a chat message using manual function-calling.
        """
        logger.info("[CHAIN_EVENT] chat start user_id=%s message=%s", user_id, message)
        logger.info("[CHAIN_EVENT] incoming chat_history_count=%s", len(chat_history) if chat_history else 0)

        await append_chat_log(user_id, "user", message)

        system_instruction_text = system_instruction
        logger.info("[CHAIN_PROMPT] final system instruction user_id=%s prompt=%s", user_id, system_instruction_text)

        config = types.GenerateContentConfig(
            tools=self.tools_list,
            system_instruction=system_instruction_text,
            temperature=0.5,
        )

        contents: List[types.Content] = []
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "assistant":
                    role = "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        logger.info("[CHAIN_EVENT] initial contents count=%s user_id=%s", len(contents), user_id)

        try:
            logger.info("[CHAIN_MODEL] first model call user_id=%s model=%s", user_id, self.model_name)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            final_response_text = ""
            max_turns = 5

            for turn in range(1, max_turns + 1):
                logger.info("[CHAIN_LOOP] turn=%s user_id=%s", turn, user_id)
                
                function_calls = []
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            function_calls.append(part.function_call)
                            
                if not function_calls:
                    final_response_text = response.text or ""
                    logger.info("[CHAIN_MODEL] no function calls turn=%s response=%s", turn, final_response_text)
                    break

                if self.verbose:
                    logger.info("[CHAIN_TOOL] function calls identified turn=%s calls=%s", turn, function_calls)

                contents.append(response.candidates[0].content)
                function_responses_parts = []

                for call in function_calls:
                    func_name = call.name
                    func_args = dict(call.args or {})
                    func_args.setdefault("user_id", user_id)
                    logger.info("[CHAIN_TOOL] invoke name=%s args=%s", func_name, func_args)

                    tool_func = self.tools_map.get(func_name)
                    if not tool_func:
                        logger.warning("[CHAIN_TOOL] missing tool name=%s", func_name)
                        function_responses_parts.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"error": "Tool not found"},
                            )
                        )
                        continue

                    try:
                        result = await tool_func(**func_args)
                        logger.info("[CHAIN_TOOL] result name=%s result=%s", func_name, result)
                        function_responses_parts.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": result},
                            )
                        )
                    except Exception as e:
                        logger.error("[CHAIN_TOOL] error name=%s error=%s", func_name, e)
                        function_responses_parts.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"error": str(e)},
                            )
                        )

                contents.append(types.Content(role="tool", parts=function_responses_parts))
                logger.info(
                    "[CHAIN_MODEL] follow-up model call after tools turn=%s tool_response_count=%s",
                    turn,
                    len(function_responses_parts),
                )
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

            if not final_response_text:
                final_response_text = response.text or ""
                logger.info("[CHAIN_MODEL] final response fallback user_id=%s response=%s", user_id, final_response_text)

            await append_chat_log(user_id, "assistant", final_response_text)
            logger.info("[CHAIN_EVENT] chat complete user_id=%s response=%s", user_id, final_response_text)
            return final_response_text

        except Exception as e:
            logger.error("[CHAIN_EVENT] chat error user_id=%s error=%s", user_id, e)
            return f"Error: {e}"
