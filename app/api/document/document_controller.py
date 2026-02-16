"""
Document Controller
Endpoints for document upload and delete.
Uses MongoMemoryClient for storage and VoyageEmbedder for vectorization.
"""

import logging
from fastapi import APIRouter, File, Form, UploadFile, Request
from typing import Optional
from datetime import datetime

from app.utils.response import success_response, error_response
from app.memory.mongo_client import MongoMemoryClient
from app.memory.embedding import VoyageEmbedder
from app.api.document.services.pdf_operation import process_pdf

logger = logging.getLogger("memory_chat.document")

router = APIRouter()

# Clients (set by main.py)
_memory_client: MongoMemoryClient = None
_embedder: VoyageEmbedder = None


def set_memory_client(client: MongoMemoryClient, embedder: VoyageEmbedder) -> None:
    """Set the memory client and embedder instance."""
    global _memory_client, _embedder
    _memory_client = client
    _embedder = embedder
    logger.info("Memory client and embedder set in document controller")


def get_memory_client() -> MongoMemoryClient:
    """Get the memory client instance."""
    if _memory_client is None:
        raise RuntimeError("Memory client not initialized")
    return _memory_client

def get_embedder() -> VoyageEmbedder:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder


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
    Upload a document to user's memory via MongoDB.
    
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
        
        # Get clients
        memory_client = get_memory_client()
        embedder = get_embedder()
        
        # Embed content
        # Note: If content is too large, we should split it. 
        # For this implementation, we assume it's chunked or we just embed the whole thing (limitations apply).
        # We'll do a simple split or just embed passing the text. Voyage handles some truncation.
        # Ideally, we should split by 1000-2000 chars.
        
        # Simple chunking for now (naive)
        chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
        
        stored_ids = []
        for i, chunk in enumerate(chunks):
            embedding = await embedder.embed_query(chunk) # or embed_documents
            doc_id = memory_client.add_memory(
                content=chunk,
                embedding=embedding,
                user_id=user_id,
                metadata={
                    "source": "document",
                    "filename": resolved_file_name,
                    "file_id": file_id,
                    "chunk_index": i
                }
            )
            stored_ids.append(doc_id)
        
        result = f"Stored {len(stored_ids)} chunks."
        logger.info(f"[UPLOAD] Storage result: {result}")
        
        return success_response(
            {
                "file_id": file_id,
                "user_id": user_id,
                "file_name": resolved_file_name,
                "file_ext": file_ext,
                "size_bytes": len(file_bytes),
                "content_length": len(content),
                "status": result,
                "chunks": len(stored_ids)
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
    
    Args:
        file_id: File identifier to delete
        user_id: User identifier
        
    Returns:
        Success response with deletion status
    """
    try:
        logger.info(f"[DELETE] Deleting file_id: {file_id}, user: {user_id}")
        
        client = get_memory_client()
        # client.collection.delete_many({"metadata.file_id": file_id, "user_id": user_id})
        # Note: We need to enable delete in MongoMemoryClient or access collection directly.
        # Accessing collection directly for now as per minimal change strategy.
        result = client.collection.delete_many({
            "user_id": user_id,
            "metadata.file_id": file_id
        })
        
        return success_response({
            "message": "Document deleted",
            "deleted_count": result.deleted_count,
            "file_id": file_id,
            "user_id": user_id
        })
        
    except Exception as e:
        logger.exception(f"[DELETE] Error: {e}")
        return error_response(f"Error deleting document: {str(e)}", 500)