"""
Logging Middleware
Request/Response logging and timing middleware.
"""

import logging
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("memory_chat.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses.
    Includes timing information and request IDs.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info("LoggingMiddleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/route handler
        
        Returns:
            HTTP Response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Log request
        start_time = time.time()
        logger.info(
            f"[{request_id}] --> {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] <-- {response.status_code} "
                f"({duration:.3f}s)"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                f"[{request_id}] !!! Error after {duration:.3f}s: {e}"
            )
            raise
