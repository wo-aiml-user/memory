"""
Application Configuration
Loads environment variables and provides configuration access.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration from environment variables."""
    
    # Gemini LLM Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Mem0 Memory Configuration
    MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")
    
    # Voyage AI Configuration (optional, for custom embeddings)
    VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
    VOYAGE_EMBED_MODEL: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3-large")
    VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # Logging Configuration
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        required = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("MEM0_API_KEY", cls.MEM0_API_KEY),
        ]
        
        missing = [name for name, value in required if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Alias for backward compatibility
settings = Config

# Create logs directory
os.makedirs(Config.LOG_DIR, exist_ok=True)
