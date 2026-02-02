"""
Document Chunking Utilities
Handles splitting large documents into Zep-compatible chunks.
"""

import logging
from typing import List

logger = logging.getLogger("memory_chat.chunking")

# Zep has a 10,000 character limit for documents
DEFAULT_CHUNK_SIZE = 9000
DEFAULT_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks under Zep's character limit.
    
    Args:
        text: The full text to chunk
        chunk_size: Maximum characters per chunk (default: 9000)
        overlap: Characters to overlap between chunks for context (default: 50)
    
    Returns:
        List of text chunks, each under chunk_size characters
    
    Example:
        >>> text = "A" * 20000  # 20k character document
        >>> chunks = chunk_text(text, chunk_size=9000, overlap=50)
        >>> len(chunks)  # 3 chunks
        3
        >>> all(len(c) <= 9000 for c in chunks)
        True
    """
    if not text:
        logger.warning("[CHUNKING] Empty text provided")
        return []
    
    text_length = len(text)
    
    # If text fits in one chunk, return as-is
    if text_length <= chunk_size:
        logger.info(f"[CHUNKING] Text fits in single chunk ({text_length} chars)")
        return [text]
    
    chunks = []
    start = 0
    chunk_number = 0
    
    while start < text_length:
        chunk_number += 1
        end = start + chunk_size
        
        # Don't exceed text length
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        logger.debug(f"[CHUNKING] Chunk {chunk_number}: chars {start}-{end} ({len(chunk)} chars)")
        
        # Move start position, accounting for overlap
        # Only apply overlap if there's more text to process
        if end < text_length:
            start = end - overlap
        else:
            break
    
    logger.info(f"[CHUNKING] Split {text_length} chars into {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={overlap})")
    
    return chunks


def chunk_text_semantic(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE
) -> List[str]:
    """
    Split text on paragraph boundaries (semantic chunking).
    Tries to keep paragraphs together when possible.
    
    Args:
        text: The full text to chunk
        max_chars: Maximum characters per chunk
    
    Returns:
        List of text chunks respecting paragraph boundaries
    """
    if not text:
        return []
    
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para_with_separator = para + "\n\n"
        
        # If adding this paragraph exceeds limit
        if len(current_chunk) + len(para_with_separator) > max_chars:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If single paragraph is too long, use character chunking
            if len(para) > max_chars:
                para_chunks = chunk_text(para, chunk_size=max_chars, overlap=50)
                chunks.extend(para_chunks)
                current_chunk = ""
            else:
                current_chunk = para_with_separator
        else:
            current_chunk += para_with_separator
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"[CHUNKING] Semantic split into {len(chunks)} chunks")
    return chunks
