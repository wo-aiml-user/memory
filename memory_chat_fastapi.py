"""
Memory Chat FastAPI
Standalone FastAPI application using Mem0 for memory.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from app.memory.mem0_client import Mem0MemoryClient
from app.memory.gemini_client import GeminiClient
from app.memory.memory_chain import MemoryChain


load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env")

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY must be set in .env")

LOG_DIR = os.getenv("LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("memory_chat")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Rotating file handler
fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "memory_chat.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
)
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

logger.info("Starting memory_chat application with Mem0")


# ============================================================================
# Initialize Services
# ============================================================================

logger.info("Initializing Gemini client")
gemini_client = GeminiClient(
    api_key=GEMINI_API_KEY,
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

logger.info("Initializing Mem0 memory client")
memory_client = Mem0MemoryClient(
    api_key=MEM0_API_KEY,
)

logger.info("Initializing memory chain")
memory_chain = MemoryChain(
    gemini_client=gemini_client,
    memory_client=memory_client,
)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Memory Chat API", version="3.0.0")


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Chat endpoint with Mem0 memory integration.
    
    Flow:
    1. Search Mem0 for relevant memories
    2. Inject memories into system prompt
    3. Generate response via Gemini
    4. Store conversation in Mem0 for future context
    """
    try:
        logger.info(f"[CHAT] user={payload.user_id}, message={payload.message[:50]}...")
        
        response = await memory_chain.chat(
            user_id=payload.user_id,
            message=payload.message,
        )
        
        return ChatResponse(
            user_id=payload.user_id,
            response=response,
        )
        
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory": "mem0",
        "llm": "gemini",
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Running FastAPI on {host}:{port}")
    uvicorn.run("memory_chat_fastapi:app", host=host, port=port, log_level="info")
