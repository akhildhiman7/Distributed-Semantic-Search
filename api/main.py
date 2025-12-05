"""
FastAPI application for semantic search.
"""
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from .config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    API_CONTACT, CORS_ORIGINS, LOG_LEVEL
)
from .models import (
    SearchRequest, SearchResponse, HealthResponse,
    StatsResponse, ErrorResponse
)
from .search_service import search_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
search_requests_total = Counter(
    'search_requests_total',
    'Total number of search requests',
    ['status']
)
search_latency_seconds = Histogram(
    'search_latency_seconds',
    'Search request latency in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
active_requests = Gauge(
    'active_requests',
    'Number of active requests'
)
collection_entities = Gauge(
    'collection_entities_total',
    'Total number of entities in collection'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting API service...")
    try:
        search_service.initialize()
        logger.info("✓ API service ready")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API service...")
    search_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Distributed Semantic Search API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats"
    }


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search for research papers",
    description="Search through 510K+ arXiv papers using natural language queries"
)
async def search(request: SearchRequest):
    """
    Perform semantic search on research papers.
    
    **Example Query:**
    ```json
    {
      "query": "neural networks for image classification",
      "top_k": 10,
      "min_score": 0.5,
      "categories": ["cs.CV", "cs.LG"]
    }
    ```
    
    **Returns:** List of relevant papers with similarity scores.
    """
    active_requests.inc()
    try:
        # Perform search
        with search_latency_seconds.time():
            results, latency_ms = search_service.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            categories=request.categories
        )
        
        # Build response
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(
            f"Search: query='{request.query[:50]}...' "
            f"results={len(results)} latency={latency_ms:.2f}ms"
        )
        
        search_requests_total.labels(status='success').inc()
        return response
        
    except ValueError as e:
        search_requests_total.labels(status='error').inc()
        search_requests_total.labels(status='error').inc()
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        search_requests_total.labels(status='error').inc()
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        search_requests_total.labels(status='error').inc()
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    finally:
        active_requests.dec()


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus metrics",
    description="Expose Prometheus metrics for monitoring"
)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    **Returns:** Metrics in Prometheus format for scraping.
    """
    # Update collection entity count
    try:
        _, _, entity_count = search_service.is_healthy()
        collection_entities.set(entity_count)
    except Exception:
        pass
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check",
    description="Check service health and Milvus connectivity"
)
async def health():
    """
    Health check endpoint.
    
    **Returns:** Service status and connection information.
    """
    try:
        milvus_connected, collection_loaded, entity_count = search_service.is_healthy()
        
        status_str = "healthy" if (milvus_connected and collection_loaded) else "unhealthy"
        
        return HealthResponse(
            status=status_str,
            milvus_connected=milvus_connected,
            collection_loaded=collection_loaded,
            total_entities=entity_count,
            model_loaded=search_service._model_loaded,
            api_version=API_VERSION,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Monitoring"],
    summary="Collection statistics",
    description="Get statistics about the indexed collection"
)
async def stats():
    """
    Get collection statistics.
    
    **Returns:** Information about the indexed papers and search configuration.
    """
    try:
        stats_data = search_service.get_collection_stats()
        
        if not stats_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stats unavailable"
            )
        
        return StatsResponse(**stats_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )
