from api.middleware import rate_limit_middleware
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import os
from typing import List

from api.models import (
    SearchRequest, InsertRequest, PaperResponse,
    HealthResponse, InsertResponse
)
from api.milvus_adapter import MilvusAdapter
from config import MilvusConfig

# Initialize FastAPI app
app = FastAPI(
    title="Paper Search API",
    description="Semantic search for academic papers using Milvus vector database",
    version="1.0.0"
)
app.middleware("http")(rate_limit_middleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
milvus_adapter = None
embedder = None

def get_milvus_config():
    """Helper function to get Milvus config"""
    return MilvusConfig()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global milvus_adapter, embedder
    
    try:
        # Initialize Milvus adapter
        config = get_milvus_config()
        milvus_adapter = MilvusAdapter(config)
        
        # Try to initialize embedder for search queries (optional)
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embedder = SentenceTransformer(model_name)
            print(f"Embedder loaded: {model_name}")
        except ImportError:
            print("sentence-transformers not installed. Search will only work with pre-computed vectors.")
            embedder = None
        except Exception as e:
            print(f"Could not load embedder: {e}")
            embedder = None
        
        print(f"API started successfully")
        print(f"   - Collection: {config.collection_name}")
        print(f"   - Vector dimension: {config.dim}")
        print(f"   - Papers in collection: {milvus_adapter.collection.num_entities}")
        
    except Exception as e:
        print("Startup failed: {e}")
        # Don't raise, let the API start without Milvus
        # Users will get errors when trying to use endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Paper Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search",
            "insert": "POST /insert",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    try:
        if milvus_adapter is None:
            return HealthResponse(
                status="unhealthy",
                milvus_connected=False,
                collection_loaded=False,
                num_papers=0,
                timestamp=datetime.now()
            )
        
        milvus_healthy = milvus_adapter.health_check()
        stats = milvus_adapter.get_stats()
        
        return HealthResponse(
            status="healthy" if milvus_healthy else "degraded",
            milvus_connected=milvus_healthy,
            collection_loaded=True,
            num_papers=stats["num_entities"],
            timestamp=datetime.now()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            milvus_connected=False,
            collection_loaded=False,
            num_papers=0,
            timestamp=datetime.now()
        )

@app.get("/stats")
async def stats():
    """Get collection statistics"""
    try:
        if milvus_adapter is None:
            raise HTTPException(status_code=503, detail="Milvus adapter not initialized")
        
        stats = milvus_adapter.get_stats()
        return {
            "collection": stats["collection_name"],
            "num_papers": stats["num_entities"],
            "has_index": stats["has_index"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[PaperResponse])
async def search(request: SearchRequest):
    """
    Search for papers similar to the query
    
    - Query is embedded using sentence-transformers
    - Results are ranked by semantic similarity
    - Optional category filtering
    """
    if milvus_adapter is None:
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
    start_time = time.time()
    
    try:
        # Generate embedding for query if embedder is available
        if embedder is None:
            raise HTTPException(
                status_code=501, 
                detail="Embedder not available. Install sentence-transformers or provide pre-computed vector."
            )
        
        query_embedding = embedder.encode(request.query).tolist()
        
        # Search in Milvus
        results = milvus_adapter.search(
            vector=query_embedding,
            top_k=request.top_k,
            categories=request.categories
        )
        
        # Filter by minimum score if provided
        if request.min_score > 0:
            results = [r for r in results if r["score"] >= request.min_score]
        
        # Convert to response model
        response = [
            PaperResponse(**paper) for paper in results
        ]
        
        # Log search metrics
        elapsed = time.time() - start_time
        print(f"🔍 Search: '{request.query[:50]}...' - {len(response)} results in {elapsed:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/insert", response_model=InsertResponse)
async def insert(request: InsertRequest):
    """
    Insert a new paper into the collection
    
    Note: Vector must be pre-computed (384-dimensional)
    """
    if milvus_adapter is None:
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
    try:
        # Validate vector dimension
        expected_dim = get_milvus_config().dim
        if len(request.vector) != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch. Expected {expected_dim}, got {len(request.vector)}"
            )
        
        # Insert paper
        inserted_count = milvus_adapter.insert_paper(
            paper_id=request.paper_id,
            title=request.title,
            abstract=request.abstract,
            categories=request.categories,
            vector=request.vector
        )
        
        return InsertResponse(
            status="success",
            paper_id=request.paper_id,
            inserted_count=inserted_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")

@app.post("/batch_insert")
async def batch_insert(papers: List[InsertRequest]):
    """
    Insert multiple papers at once
    
    Note: All vectors must be pre-computed
    """
    if milvus_adapter is None:
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
    try:
        # Convert to dictionary format
        paper_dicts = []
        for paper in papers:
            paper_dicts.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "categories": paper.categories,
                "vector": paper.vector
            })
        
        # Batch insert
        total_inserted = milvus_adapter.batch_insert(paper_dicts)
        
        return {
            "status": "success",
            "total_inserted": total_inserted,
            "message": f"Successfully inserted {total_inserted} papers"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__}
    )

# Add prometheus metrics endpoint if needed
@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    return {
        "timestamp": datetime.now().isoformat(),
        "service": "paper-search-api",
        "version": "1.0.0"
    }