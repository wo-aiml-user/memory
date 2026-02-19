"""
Memory Chain
Main orchestration logic for memory-enhanced chat.
Combines Gemini LLM with Zep memory.

Flow: User Message → Get Context from Zep → Inject into Prompt → Gemini → Response → Store in Zep
"""

import logging
from typing import Optional
from datetime import datetime

from .gemini_client import GeminiClient
from .zep_client import ZepMemoryClient
from .prompt import SYSTEM_PROMPT

logger = logging.getLogger("memory_chat.chain")


class MemoryChain:
    """
    Orchestrates memory-enhanced chat using Gemini LLM and Zep memory.
    
    Memory Flow:
    1. User sends message
    2. Get context from Zep (includes summary + relevant facts)
    3. Inject context into system prompt
    4. Gemini generates response
    5. Store conversation in Zep
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        memory_client: ZepMemoryClient,
        auto_store_memory: bool = True,
    ):
        """
        Initialize the memory chain.
        
        Args:
            gemini_client: Gemini LLM client
            memory_client: Zep memory client
            auto_store_memory: Automatically store conversations after response
        """
        self.llm = gemini_client
        self.memory = memory_client
        self.auto_store_memory = auto_store_memory
        
        logger.info("MemoryChain initialized (auto_store_memory=%s)", auto_store_memory)
    
    async def chat(self, user_id: str, message: str) -> str:
        """
        Process a chat message with memory context injection.
        
        Flow:
        1. Get context from Zep
        2. Inject context into prompt
        3. Gemini generates response
        4. Store conversation async
        
        Args:
            user_id: User identifier
            message: User's message
        
        Returns:
            Assistant's response
        """
        logger.info("MEMORY CHAIN: PROCESSING CHAT MESSAGE")        
        try:
            # ============================================================
            # STEP 1: LOG THE INCOMING MESSAGE
            # ============================================================
            logger.info(f"  User ID: {user_id}")
            logger.info(f"  Message: {message}")            
            # ============================================================
            # STEP 2: GET CONTEXT FROM ZEP
            # ============================================================
            context = await self.memory.get_context(user_id)
            
            if context:
                logger.info(f"  Context retrieved from Zep!")
                logger.info(f"  Context Length: {len(context)} characters")
                logger.info("  FULL CONTEXT FROM ZEP:")
                for line in context.split('\n'):
                    logger.info(f"  | {line}")
            else:
                logger.info("No context available (new user or no history)")
            
            # ============================================================
            # STEP 3: BUILD PROMPT WITH CONTEXT INJECTION
            # ============================================================
            logger.info("BUILDING PROMPT WITH CONTEXT INJECTION")
            
            if context:
                system_prompt = f"{SYSTEM_PROMPT}\n\n<USER_CONTEXT>\n{context}\n</USER_CONTEXT>"
                logger.info(f" Context INJECTED into system prompt")
            else:
                system_prompt = SYSTEM_PROMPT
                logger.info(f" No context to inject (using base prompt only)")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            logger.info(f"  System Prompt Length: {len(system_prompt)} characters")
            logger.info(f"  Total messages to LLM: {len(messages)}")
            logger.info("  FULL PROMPT BEING SENT TO GEMINI:")
            for msg in messages:
                logger.info(f"  [{msg['role'].upper()}]:")
                for line in msg['content'].split('\n'):
                    logger.info(f"  | {line}")
            
            # ============================================================
            # STEP 4: CALL GEMINI LLM
            # ============================================================
            logger.info(f"  Sending request to Gemini...")
            
            start_time = datetime.now()
            response = self.llm.chat_completion(messages)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            assistant_response = response.choices[0].message.content or ""
            
            # Log token usage if available
            usage = self.llm.get_usage(response)
            
            logger.info(f"  Response received from Gemini!")
            logger.info(f"  Response Time: {elapsed:.2f}s")
            logger.info(f"  Token Usage: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
            logger.info(f"  Response Length: {len(assistant_response)} characters")
            logger.info("  FULL LLM RESPONSE:")
            for line in assistant_response.split('\n'):
                logger.info(f"  | {line}")
            
            # ============================================================
            # STEP 5: STORE CONVERSATION IN ZEP
            # ============================================================
            if self.auto_store_memory:
                logger.info("STORING CONVERSATION IN ZEP")
                logger.info(f"  Thread ID: thread_{user_id}")
                logger.info(f"  [USER MESSAGE]:")
                for line in message.split('\n'):
                    logger.info(f"  | {line}")
                logger.info(f"  [ASSISTANT RESPONSE]:")
                for line in assistant_response.split('\n'):
                    logger.info(f"  | {line}")
                
                await self.memory.add_messages(
                    user_id=user_id,
                    user_message=message,
                    assistant_response=assistant_response,
                    return_context=False
                )
                
                logger.info(f"  Conversation stored in Zep - will be processed for facts/entities")
            else:
                logger.info("SKIPPING STORAGE (auto_store_memory=False)")
            
            logger.info("MEMORY CHAIN: COMPLETED SUCCESSFULLY")
            
            return assistant_response
            
        except Exception as e:
            logger.exception(f"[ERROR] Memory chain failed: {e}")
            raise
