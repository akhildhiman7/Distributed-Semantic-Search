"""
Search service for encoding queries and searching Milvus.
"""
import logging
import time
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from .config import (
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
    MODEL_NAME, SEARCH_PARAMS
)
from .models import PaperResult

logger = logging.getLogger(__name__)


class SearchService:
    """Service for semantic search operations."""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.collection: Optional[Collection] = None
        self._model_loaded = False
        self._milvus_connected = False
        
    def initialize(self):
        """Initialize model and Milvus connection."""
        logger.info("Initializing search service...")
        
        # Load SentenceTransformer model
        try:
            logger.info(f"Loading model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)
            # Warmup
            _ = self.model.encode(["warmup"], show_progress_bar=False)
            self._model_loaded = True
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Connect to Milvus
        try:
            logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            self._milvus_connected = True
            logger.info("✓ Connected to Milvus")
            
            # Load collection
            if utility.has_collection(MILVUS_COLLECTION):
                self.collection = Collection(MILVUS_COLLECTION)
                self.collection.load()
                logger.info(f"✓ Collection '{MILVUS_COLLECTION}' loaded")
            else:
                raise ValueError(f"Collection '{MILVUS_COLLECTION}' not found")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def is_healthy(self) -> Tuple[bool, bool, int]:
        """
        Check service health.
        
        Returns:
            Tuple of (milvus_connected, collection_loaded, entity_count)
        """
        try:
            if not self._milvus_connected or self.collection is None:
                return False, False, 0
            
            # Check if collection is loaded
            if not self.collection:
                return self._milvus_connected, False, 0
            
            # Get entity count
            entity_count = self.collection.num_entities
            
            return True, True, entity_count
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False, False, 0
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text to embedding vector.
        
        Args:
            query: Search query text
            
        Returns:
            Normalized embedding vector
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: Optional[float] = None,
        categories: Optional[List[str]] = None
    ) -> Tuple[List[PaperResult], float]:
        """
        Perform semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            categories: Filter by categories
            
        Returns:
            Tuple of (results, latency_ms)
        """
        start_time = time.time()
        
        # Encode query
        try:
            query_embedding = self.encode_query(query)
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise ValueError(f"Query encoding failed: {str(e)}")
        
        # Build filter expression
        expr = None
        if categories:
            # Filter by categories
            category_filters = " or ".join([
                f'categories like "%{cat}%"' for cat in categories
            ])
            expr = f"({category_filters})"
        
        # Search in Milvus
        try:
            search_params = SEARCH_PARAMS.copy()
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2 if min_score else top_k,  # Get more if filtering by score
                expr=expr,
                output_fields=[
                    "paper_id", "title", "abstract",
                    "categories", "text_length", "has_full_data"
                ]
            )
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
        
        # Process results
        paper_results = []
        for hits in results:
            for hit in hits:
                # Apply score threshold
                if min_score and hit.score < min_score:
                    continue
                
                paper_results.append(PaperResult(
                    paper_id=str(hit.entity.get("paper_id")),
                    title=hit.entity.get("title"),
                    abstract=hit.entity.get("abstract"),
                    categories=hit.entity.get("categories"),
                    score=round(float(hit.score), 4),
                    text_length=int(hit.entity.get("text_length")),
                    has_full_data=bool(hit.entity.get("has_full_data"))
                ))
                
                # Stop if we have enough results
                if len(paper_results) >= top_k:
                    break
        
        latency_ms = (time.time() - start_time) * 1000
        
        return paper_results[:top_k], latency_ms
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        if not self.collection:
            return {}
        
        try:
            # Get collection info
            entity_count = self.collection.num_entities
            
            # Get index info
            indexes = self.collection.indexes
            index_type = "Unknown"
            if indexes:
                index_type = indexes[0].params.get("index_type", "Unknown")
            
            return {
                "collection_name": MILVUS_COLLECTION,
                "total_entities": entity_count,
                "index_type": index_type,
                "vector_dim": 384,
                "metric_type": "IP",
                "model_name": MODEL_NAME
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down search service...")
        try:
            if self.collection:
                self.collection.release()
            if self._milvus_connected:
                connections.disconnect("default")
            logger.info("✓ Search service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global service instance
search_service = SearchService()
