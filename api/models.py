from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    categories: Optional[List[str]] = None
    min_score: Optional[float] = 0.0

class InsertRequest(BaseModel):
    paper_id: int  # Note: INT64 per his schema
    title: str
    abstract: str
    categories: List[str]
    vector: List[float]  # Pre-computed embedding

class PaperResponse(BaseModel):
    paper_id: int
    title: str
    abstract: str
    categories: List[str]
    score: float

class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    collection_loaded: bool
    num_papers: int
    timestamp: datetime

class InsertResponse(BaseModel):
    status: str
    paper_id: int
    inserted_count: int