"""
Memory Chat FastAPI
Standalone FastAPI application using Zep Cloud for memory.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from app.memory.zep_client import ZepMemoryClient
from app.memory.gemini_client import GeminiClient
from app.memory.memory_chain import MemoryChain


load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env")

ZEP_API_KEY = os.getenv("ZEP_API_KEY")
if not ZEP_API_KEY:
    raise ValueError("ZEP_API_KEY must be set in .env")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

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

logger.info("Starting memory_chat application with Zep Cloud")


# ============================================================================
# Initialize Services
# ============================================================================

logger.info("Initializing Gemini client")
gemini_client = GeminiClient(
    api_key=GEMINI_API_KEY,
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

logger.info("Initializing Zep memory client")
memory_client = ZepMemoryClient(
    zep_api_key=ZEP_API_KEY,
    voyage_api_key=VOYAGE_API_KEY,
    voyage_embed_model=os.getenv("VOYAGE_EMBED_MODEL", "voyage-3-large"),
    voyage_rerank_model=os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5"),
)

logger.info("Initializing memory chain")
memory_chain = MemoryChain(
    gemini_client=gemini_client,
    memory_client=memory_client,
)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Memory Chat API")


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Chat endpoint with Zep memory integration.
    
    Flow:
    1. Get context from Zep (user summary + facts)
    2. Inject context into system prompt
    3. Generate response via Gemini
    4. Store conversation in Zep for future context
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
        "memory": "zep",
        "llm": "gemini",
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Running FastAPI on {host}:{port}")
    uvicorn.run("memory_chat_fastapi:app", host=host, port=port, log_level="info")
