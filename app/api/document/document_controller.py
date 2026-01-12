"""
Document Controller
Endpoints for document upload and delete.
Uses Mem0 for memory storage.
"""

import logging
from fastapi import APIRouter, File, Form, UploadFile, Request
from typing import Optional
from datetime import datetime

from app.utils.response import success_response, error_response
from app.memory.mem0_client import Mem0MemoryClient
from app.api.document.services.pdf_operation import process_pdf

logger = logging.getLogger("memory_chat.document")

router = APIRouter()

# Memory client instance (set by main.py)
_memory_client: Mem0MemoryClient = None


def set_memory_client(client: Mem0MemoryClient) -> None:
    """Set the memory client instance."""
    global _memory_client
    _memory_client = client
    logger.info("Memory client set in document controller")


def get_memory_client() -> Mem0MemoryClient:
    """Get the memory client instance."""
    if _memory_client is None:
        raise RuntimeError("Memory client not initialized")
    return _memory_client


def extract_text_from_file(file_id: str, file_bytes: bytes, file_ext: str) -> str:
    """
    Extract text content from uploaded file.
    Uses Apache Tika for PDF extraction.
    """
    try:
        if file_ext == "pdf":
            # Use Tika for PDF extraction
            logger.info(f"[EXTRACT] Using Tika for PDF: {file_id}")
            page_texts = process_pdf(file_id, file_bytes)
            content = "\n\n".join(page_texts)
            logger.info(f"[EXTRACT] Extracted {len(content)} characters from PDF")
            return content
        elif file_ext in ["txt", "md", "json", "csv"]:
            return file_bytes.decode("utf-8")
        elif file_ext in ["doc", "docx"]:
            # TODO: Add DOCX support via Tika
            return f"[DOCX content - {len(file_bytes)} bytes]"
        else:
            return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception(f"Error extracting text: {e}")
        raise


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    file_id: str = Form(...),
    user_id: str = Form(...),
    file_name: Optional[str] = Form(None),
):
    """
    Upload a document to user's memory via Mem0.
    
    Args:
        file: The file to upload
        file_id: Unique identifier for the file
        user_id: User identifier
        file_name: Optional override for filename
        
    Returns:
        Success response with storage status
    """
    try:
        logger.info(f"[UPLOAD] Starting for file_id: {file_id}, user: {user_id}")
        
        # Read file
        file_bytes = await file.read()
        resolved_file_name = file_name or file.filename
        file_ext = resolved_file_name.split(".")[-1].lower() if "." in resolved_file_name else ""
        
        logger.info(f"[UPLOAD] File: {resolved_file_name}, Size: {len(file_bytes)} bytes")
        
        # Extract text (uses Tika for PDF)
        content = extract_text_from_file(file_id, file_bytes, file_ext)
        logger.info(f"[UPLOAD] Extracted {len(content)} characters")
        
        # Get memory client
        memory_client = get_memory_client()
        
        # Add to Mem0 as document memory
        result = await memory_client.add_messages(
            user_id=user_id,
            user_message=f"I'm uploading a document: {resolved_file_name}",
            assistant_response=f"Document content from {resolved_file_name}:\n\n{content}",
        )
        
        logger.info(f"[UPLOAD] Storage result: {result}")
        
        return success_response(
            {
                "file_id": file_id,
                "user_id": user_id,
                "file_name": resolved_file_name,
                "file_ext": file_ext,
                "size_bytes": len(file_bytes),
                "content_length": len(content),
                "status": result
            },
            201,
        )
        
    except RuntimeError as e:
        logger.error(f"[UPLOAD] Service not ready: {e}")
        return error_response("Document service not initialized", 503)
    except Exception as e:
        logger.exception(f"[UPLOAD] Error: {e}")
        return error_response(f"Error uploading document: {str(e)}", 500)


@router.delete("/delete")
async def delete_document(
    request: Request,
    file_id: str = Form(...),
    user_id: str = Form(...),
):
    """
    Delete a document from memory.
    
    Note: Mem0 deletion requires memory_id which we don't track yet.
    
    Args:
        file_id: File identifier to delete
        user_id: User identifier
        
    Returns:
        Success response with deletion status
    """
    try:
        logger.info(f"[DELETE] Deleting file_id: {file_id}, user: {user_id}")
        
        # TODO: Implement Mem0 memory deletion (requires tracking memory_id)
        return success_response({
            "message": "Document deletion not yet supported",
            "file_id": file_id,
            "user_id": user_id
        })
        
    except Exception as e:
        logger.exception(f"[DELETE] Error: {e}")
        return error_response(f"Error deleting document: {str(e)}", 500)