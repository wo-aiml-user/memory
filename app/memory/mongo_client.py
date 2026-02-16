"""
MongoDB Memory Client
Handles storage and retrieval of memories using MongoDB and Vector Search.
"""

import logging
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection

# Reuse existing embedder
from .embedding import VoyageEmbedder

logger = logging.getLogger("memory_chat.mongo_client")

class MongoMemoryClient:
    """
    Client for interacting with MongoDB for memory storage and retrieval.
    Supports vector search using numpy for similarity (compatible with standard Mongo).
    """
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "memory", collection_name: str = "user-memory"):
        """
        Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        
        # Ensure index on user_id for faster filtering
        self.collection.create_index("user_id")
        
        logger.info(f"Connected to MongoDB: {db_name}.{collection_name}")

    def add_memory(self, content: str, embedding: List[float], user_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a memory to the database.
        
        Args:
            content: The text content of the memory
            embedding: Vector embedding of the content
            user_id: User identifier (owner of the memory)
            metadata: Additional metadata
            
        Returns:
            Inserted document ID as string
        """
        document = {
            "content": content,
            "embedding": embedding,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow()
        }
        
        result = self.collection.insert_one(document)
        doc_id = str(result.inserted_id)
        logger.info(f"Stored memory {doc_id} for user {user_id}")
        return doc_id

    def search_memories(self, query_embedding: List[float], user_id: str, limit: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using cosine similarity.
        
        Args:
            query_embedding: Embedding of the query
            user_id: User identifier to filter by
            limit: Maximum number of results
            threshold: Minimum similarity score
            
        Returns:
            List of memory documents with similarity score
        """
        # 1. Fetch all memories for the user
        # Note: For production with millions of records, use Atlas Search ($vectorSearch).
        # For this "custom architecture" without assuming Atlas, we fetch and compute locally.
        cursor = self.collection.find({"user_id": user_id})
        candidates = list(cursor)
        
        if not candidates:
            return []
            
        # 2. Calculate Cosine Similarity
        results = []
        query_vec = np.array(query_embedding)
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0:
            return []
            
        for doc in candidates:
            doc_vec = np.array(doc["embedding"])
            norm_doc = np.linalg.norm(doc_vec)
            
            if norm_doc == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                
            if similarity >= threshold:
                # Add similarity to doc and remove raw embedding from result to save bandwidth
                doc["similarity"] = float(similarity)
                # doc.pop("embedding", None) # Optional: keep or remove
                results.append(doc)
                
        # 3. Sort and Limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def close(self):
        self.client.close()
