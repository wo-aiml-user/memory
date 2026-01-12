"""
Memory Chain
Main orchestration logic for memory-enhanced chat.
Combines Gemini LLM with Mem0 memory.

Flow: User Message -> Search Mem0 for Context -> Inject into Prompt -> Gemini -> Response -> Store in Mem0
"""

import logging
from typing import Optional
from datetime import datetime

from .gemini_client import GeminiClient
from .mem0_client import Mem0MemoryClient
from .prompt import SYSTEM_PROMPT

logger = logging.getLogger("memory_chat.chain")


class MemoryChain:
    """
    Orchestrates memory-enhanced chat using Gemini LLM and Mem0 memory.
    
    Memory Flow:
    1. User sends message
    2. Search Mem0 for relevant memories (using message as query)
    3. Inject memories into system prompt
    4. Gemini generates response
    5. Store conversation in Mem0
    """
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        memory_client: Mem0MemoryClient,
        auto_store_memory: bool = True,
    ):
        """
        Initialize the memory chain.
        
        Args:
            gemini_client: Gemini LLM client
            memory_client: Mem0 memory client
            auto_store_memory: Automatically store conversations after response
        """
        self.llm = gemini_client
        self.memory = memory_client
        self.auto_store_memory = auto_store_memory
        
        logger.info("MemoryChain initialized with Mem0 (auto_store_memory=%s)", auto_store_memory)
    
    async def chat(self, user_id: str, message: str) -> str:
        """
        Process a chat message with memory context injection.
        
        Flow:
        1. Search Mem0 for relevant memories
        2. Inject memories into prompt
        3. Gemini generates response
        4. Store conversation in Mem0
        
        Args:
            user_id: User identifier
            message: User's message
        
        Returns:
            Assistant's response
        """
        logger.info("=" * 80)
        logger.info("MEMORY CHAIN: PROCESSING CHAT MESSAGE")
        logger.info("=" * 80)
        
        try:
            # ============================================================
            # STEP 1: LOG THE INCOMING MESSAGE
            # ============================================================
            logger.info("[STEP 1/5] RECEIVED USER MESSAGE")
            logger.info(f"  User ID: {user_id}")
            logger.info(f"  Message: {message}")
            logger.info(f"  Message Length: {len(message)} characters")
            
            # ============================================================
            # STEP 2: SEARCH MEM0 FOR RELEVANT MEMORIES
            # ============================================================
            logger.info("[STEP 2/5] SEARCHING MEM0 FOR RELEVANT MEMORIES")
            logger.info(f"  Search query: {message}")
            context = await self.memory.get_context(user_id, message)
            
            if context:
                logger.info(f"  [OK] Found relevant memories!")
                logger.info(f"  Context Length: {len(context)} characters")
                logger.info("  " + "-" * 60)
                logger.info("  MEMORIES FROM MEM0:")
                logger.info("  " + "-" * 60)
                for line in context.split('\n'):
                    logger.info(f"  | {line}")
                logger.info("  " + "-" * 60)
            else:
                logger.info("  [X] No context available (new user or no history)")
            
            # ============================================================
            # STEP 3: BUILD PROMPT WITH CONTEXT INJECTION
            # ============================================================
            logger.info("[STEP 3/5] BUILDING PROMPT WITH CONTEXT INJECTION")
            
            if context:
                system_prompt = f"{SYSTEM_PROMPT}\n\n<USER_CONTEXT>\n{context}\n</USER_CONTEXT>"
                logger.info(f"  [OK] Context INJECTED into system prompt")
            else:
                system_prompt = SYSTEM_PROMPT
                logger.info(f"  [!] No context to inject (using base prompt only)")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            logger.info(f"  System Prompt Length: {len(system_prompt)} characters")
            logger.info(f"  Total messages to LLM: {len(messages)}")
            logger.info("  " + "-" * 60)
            logger.info("  FULL PROMPT BEING SENT TO GEMINI:")
            logger.info("  " + "-" * 60)
            for msg in messages:
                logger.info(f"  [{msg['role'].upper()}]:")
                for line in msg['content'].split('\n'):
                    logger.info(f"  | {line}")
                logger.info("  " + "-" * 40)
            logger.info("  " + "-" * 60)
            
            # ============================================================
            # STEP 4: CALL GEMINI LLM
            # ============================================================
            logger.info("[STEP 4/5] CALLING GEMINI LLM")
            logger.info(f"  Sending request to Gemini...")
            
            start_time = datetime.now()
            response = self.llm.chat_completion(messages)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            assistant_response = response.choices[0].message.content or ""
            
            # Log token usage if available
            usage = self.llm.get_usage(response)
            
            logger.info(f"  [OK] Response received from Gemini!")
            logger.info(f"  Response Time: {elapsed:.2f}s")
            logger.info(f"  Token Usage: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
            logger.info(f"  Response Length: {len(assistant_response)} characters")
            logger.info("  " + "-" * 60)
            logger.info("  FULL LLM RESPONSE:")
            logger.info("  " + "-" * 60)
            for line in assistant_response.split('\n'):
                logger.info(f"  | {line}")
            logger.info("  " + "-" * 60)
            
            # ============================================================
            # STEP 5: STORE CONVERSATION IN ZEP
            # ============================================================
            if self.auto_store_memory:
                logger.info("[STEP 5/5] STORING CONVERSATION IN ZEP")
                logger.info("  " + "-" * 60)
                logger.info("  DATA BEING SENT TO MEM0 FOR STORAGE:")
                logger.info("  " + "-" * 60)
                logger.info(f"  Thread ID: thread_{user_id}")
                logger.info(f"  [USER MESSAGE]:")
                for line in message.split('\n'):
                    logger.info(f"  | {line}")
                logger.info(f"  [ASSISTANT RESPONSE]:")
                for line in assistant_response.split('\n'):
                    logger.info(f"  | {line}")
                logger.info("  " + "-" * 60)
                
                await self.memory.add_messages(
                    user_id=user_id,
                    user_message=message,
                    assistant_response=assistant_response,
                )
                
                logger.info(f"  [OK] Conversation stored in Mem0 - will be processed for facts/entities")
            else:
                logger.info("[STEP 5/5] SKIPPING STORAGE (auto_store_memory=False)")
            
            logger.info("=" * 80)
            logger.info("MEMORY CHAIN: COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return assistant_response
            
        except Exception as e:
            logger.exception(f"[ERROR] Memory chain failed: {e}")
            raise
