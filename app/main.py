"""
Memory Chat API - FastAPI Entry Point
Main application initialization and configuration.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Config
from app.middleware.logging import LoggingMiddleware
from app.middleware.jwt_auth import JWTAuthMiddleware
from app.api.chat.chat_controller import set_chat_service
from app.api.chat.services.chat_service import ChatService
from app.api.document.document_controller import set_memory_client as set_document_memory_client
from app.memory.memory_chain import MemoryChain
from app.memory.gemini_client import GeminiClient
from app.memory.supermemory_client import SupermemoryClient
from app.route import setup_routes


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure application logging."""
    
    logger = logging.getLogger("memory_chat")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.DEBUG))
    
    # Create logs directory
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        os.path.join(Config.LOG_DIR, "memory_chat.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# Application Components
# ============================================================================

# Global instances
_memory_client: SupermemoryClient = None
_gemini_client: GeminiClient = None
_memory_chain: MemoryChain = None
_chat_service: ChatService = None


async def initialize_services():
    """Initialize all application services."""
    global _memory_client, _gemini_client, _memory_chain, _chat_service
    
    logger.info("Initializing services...")
    
    try:
        # Initialize Gemini client (for chat)
        logger.info("Initializing Gemini client")
        _gemini_client = GeminiClient(
            api_key=Config.GEMINI_API_KEY,
            model=Config.GEMINI_MODEL,
        )
        
        # Initialize Supermemory client
        logger.info("Initializing Supermemory client")
        _memory_client = SupermemoryClient(
            api_key=Config.SUPERMEMORY_API_KEY,
        )
        
        # Initialize memory chain
        logger.info("Initializing memory chain")
        _memory_chain = MemoryChain(
            gemini_client=_gemini_client,
            memory_client=_memory_client,
        )
        
        # Initialize chat service
        logger.info("Initializing chat service")
        _chat_service = ChatService(memory_chain=_memory_chain)
        
        # Set service in controller
        set_chat_service(_chat_service)
        
        # Set memory client in document controller
        set_document_memory_client(_memory_client)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.exception(f"Error initializing services: {e}")
        raise


async def shutdown_services():
    """Cleanup services on shutdown."""
    global _memory_client
    
    logger.info("Shutting down services...")
    
    try:
        if _memory_client:
            await _memory_client.close()
            logger.info("Memory client closed")
    except Exception as e:
        logger.exception(f"Error during shutdown: {e}")


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Memory Chat API...")
    await initialize_services()
    yield
    logger.info("Shutting down Memory Chat API...")
    await shutdown_services()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Memory Chat API",
    description="Chat API with Supermemory",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Middleware
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# JWT Auth middleware (disabled by default)
app.add_middleware(JWTAuthMiddleware, enabled=False)


# ============================================================================
# Routes
# ============================================================================

setup_routes(app)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "memory": "supermemory",
        "llm": "gemini",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Memory Chat API",
        "version": "2.0.0",
        "memory_backend": "Supermemory",
    }
