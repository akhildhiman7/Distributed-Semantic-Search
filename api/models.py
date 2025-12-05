"""
Pydantic models for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language search query",
        example="neural networks for image classification"
    )
    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of results to return",
        example=10
    )
    min_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0-1.0)",
        example=0.5
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Filter by arXiv categories (e.g., ['cs.LG', 'cs.AI'])",
        example=["cs.LG", "cs.AI"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "deep learning for natural language processing",
                "top_k": 5,
                "min_score": 0.6,
                "categories": ["cs.CL", "cs.LG"]
            }
        }


class PaperResult(BaseModel):
    """Single paper result."""
    paper_id: str = Field(..., description="ArXiv paper ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    categories: str = Field(..., description="ArXiv categories (space-separated)")
    score: float = Field(..., description="Similarity score (0.0-1.0)", ge=0.0, le=1.0)
    text_length: int = Field(..., description="Combined text length")
    has_full_data: bool = Field(..., description="Data completeness flag")

    class Config:
        json_schema_extra = {
            "example": {
                "paper_id": "200303253",
                "title": "Introduction to deep learning",
                "abstract": "Deep Learning has made a major impact on data science...",
                "categories": "cs.LG",
                "score": 0.6478,
                "text_length": 5234,
                "has_full_data": True
            }
        }


class SearchResponse(BaseModel):
    """Search response model."""
    query: str = Field(..., description="Original search query")
    results: List[PaperResult] = Field(..., description="List of matching papers")
    total_results: int = Field(..., description="Number of results returned")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "neural networks deep learning",
                "results": [
                    {
                        "paper_id": "200303253",
                        "title": "Introduction to deep learning",
                        "abstract": "Deep Learning has made a major impact...",
                        "categories": "cs.LG",
                        "score": 0.6478,
                        "text_length": 5234,
                        "has_full_data": True
                    }
                ],
                "total_results": 1,
                "latency_ms": 218.5,
                "timestamp": "2025-12-05T00:53:07Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    milvus_connected: bool = Field(..., description="Milvus connection status")
    collection_loaded: bool = Field(..., description="Collection load status")
    total_entities: int = Field(..., description="Total papers indexed")
    model_loaded: bool = Field(..., description="Embedding model status")
    api_version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class StatsResponse(BaseModel):
    """Statistics response."""
    collection_name: str = Field(..., description="Milvus collection name")
    total_entities: int = Field(..., description="Total papers indexed")
    index_type: str = Field(..., description="Vector index type")
    vector_dim: int = Field(..., description="Embedding dimension")
    metric_type: str = Field(..., description="Similarity metric")
    model_name: str = Field(..., description="Embedding model name")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Query must be at least 3 characters",
                "timestamp": "2025-12-05T00:53:07Z"
            }
        }
