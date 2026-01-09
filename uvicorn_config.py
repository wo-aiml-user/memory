"""
Uvicorn Server Configuration
Configuration for running the FastAPI server with uvicorn.
"""

import os
from app.config import Config

# Server configuration
HOST = Config.HOST
PORT = Config.PORT

# Uvicorn settings
UVICORN_CONFIG = {
    "app": "app.main:app",
    "host": HOST,
    "port": PORT,
    "reload": True,
    "reload_dirs": ["app", "tools"],
    "log_level": "info",
    "access_log": True,
    "workers": 1,  # Use 1 worker for development, increase for production
}

# Production settings (use these in production)
PRODUCTION_CONFIG = {
    "app": "app.main:app",
    "host": HOST,
    "port": PORT,
    "reload": False,
    "log_level": "warning",
    "access_log": True,
    "workers": 4,  # Adjust based on CPU cores
    "limit_concurrency": 100,
    "limit_max_requests": 10000,
    "timeout_keep_alive": 5,
}


if __name__ == "__main__":
    import uvicorn
    
    # Use development config by default
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        config = PRODUCTION_CONFIG
    else:
        config = UVICORN_CONFIG
    
    uvicorn.run(**config)
