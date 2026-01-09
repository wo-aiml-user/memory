"""
Response Utilities
Standard response formatting for API endpoints.
"""

from fastapi.responses import JSONResponse
from typing import Any, Optional


def success_response(data: Any = None, status_code: int = 200) -> JSONResponse:
    """
    Create a successful JSON response.
    
    Args:
        data: Response data
        status_code: HTTP status code
    
    Returns:
        JSONResponse with success format
    """
    content = {
        "success": True,
        "data": data
    }
    return JSONResponse(content=content, status_code=status_code)


def error_response(error: str, status_code: int = 400) -> JSONResponse:
    """
    Create an error JSON response.
    
    Args:
        error: Error message
        status_code: HTTP status code
    
    Returns:
        JSONResponse with error format
    """
    content = {
        "success": False,
        "error": error
    }
    return JSONResponse(content=content, status_code=status_code)
