from api.middleware import rate_limit_middleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import os
import psutil
from typing import List, Dict

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

# Metrics collection
class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.request_counts = {
            "total": 0,
            "search": 0,
            "insert": 0,
            "health": 0,
            "stats": 0,
            "metrics": 0,
            "errors": 0
        }
        self.search_stats = {
            "total_searches": 0,
            "total_results_returned": 0,
            "avg_search_time": 0.0,
            "search_times": []
        }

    def increment_request(self, endpoint: str):
        self.request_counts["total"] += 1
        if endpoint in self.request_counts:
            self.request_counts[endpoint] += 1

    def increment_error(self):
        self.request_counts["errors"] += 1

    def record_search(self, num_results: int, search_time: float):
        self.search_stats["total_searches"] += 1
        self.search_stats["total_results_returned"] += num_results
        self.search_stats["search_times"].append(search_time)

        # Keep only last 100 search times
        if len(self.search_stats["search_times"]) > 100:
            self.search_stats["search_times"] = self.search_stats["search_times"][-100:]

        # Update average
        if self.search_stats["search_times"]:
            self.search_stats["avg_search_time"] = sum(self.search_stats["search_times"]) / len(self.search_stats["search_times"])

# Global metrics collector
metrics_collector = MetricsCollector()

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
            print(f"✅ Embedder loaded: {model_name}")
        except ImportError:
            print("⚠️  sentence-transformers not installed. Search will only work with pre-computed vectors.")
            embedder = None
        except Exception as e:
            print(f"⚠️  Could not load embedder: {e}")
            embedder = None
        
        print(f"✅ API started successfully")
        print(f"   - Collection: {config.collection_name}")
        print(f"   - Vector dimension: {config.dim}")
        print(f"   - Papers in collection: {milvus_adapter.collection.num_entities}")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # Don't raise, let the API start without Milvus
        # Users will get errors when trying to use endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    metrics_collector.increment_request("root")
    return {
        "message": "Paper Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search",
            "insert": "POST /insert",
            "health": "GET /health",
            "stats": "GET /stats",
            "metrics": "GET /metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    metrics_collector.increment_request("health")
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
        metrics_collector.increment_error()
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
    metrics_collector.increment_request("stats")
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
        metrics_collector.increment_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[PaperResponse])
async def search(request: SearchRequest):
    """
    Search for papers similar to the query
    
    - Query is embedded using sentence-transformers
    - Results are ranked by semantic similarity
    - Optional category filtering
    """
    metrics_collector.increment_request("search")

    if milvus_adapter is None:
        metrics_collector.increment_error()
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
    start_time = time.time()
    
    try:
        # Generate embedding for query if embedder is available
        if embedder is None:
            metrics_collector.increment_error()
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
        metrics_collector.record_search(len(response), elapsed)

        print(f"🔍 Search: '{request.query[:50]}...' - {len(response)} results in {elapsed:.3f}s")
        
        return response
        
    except HTTPException:
        metrics_collector.increment_error()
        raise
    except Exception as e:
        metrics_collector.increment_error()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/insert", response_model=InsertResponse)
async def insert(request: InsertRequest):
    """
    Insert a new paper into the collection
    
    Note: Vector must be pre-computed (384-dimensional)
    """
    metrics_collector.increment_request("insert")

    if milvus_adapter is None:
        metrics_collector.increment_error()
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
    try:
        # Validate vector dimension
        expected_dim = get_milvus_config().dim
        if len(request.vector) != expected_dim:
            metrics_collector.increment_error()
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
        metrics_collector.increment_error()
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")

@app.post("/batch_insert")
async def batch_insert(papers: List[InsertRequest]):
    """
    Insert multiple papers at once
    
    Note: All vectors must be pre-computed
    """
    metrics_collector.increment_request("insert")

    if milvus_adapter is None:
        metrics_collector.increment_error()
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
        metrics_collector.increment_error()
        raise HTTPException(status_code=500, detail=f"Batch insert failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring and observability"""
    metrics_collector.increment_request("metrics")

    try:
        # System metrics
        process = psutil.Process(os.getpid())
        system_memory = psutil.virtual_memory()

        # API uptime
        uptime_seconds = time.time() - metrics_collector.start_time
        uptime_hours = uptime_seconds / 3600

        # Milvus metrics if available
        milvus_metrics = {}
        if milvus_adapter:
            try:
                stats = milvus_adapter.get_stats()
                milvus_metrics = {
                    "collection_name": stats["collection_name"],
                    "num_entities": stats["num_entities"],
                    "has_index": stats["has_index"]
                }
            except:
                milvus_metrics = {"error": "Could not fetch Milvus metrics"}

        # Build response
        response = {
            "timestamp": datetime.now().isoformat(),
            "service": "paper-search-api",
            "version": "1.0.0",

            # System metrics
            "system": {
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_hours": round(uptime_hours, 2),
                "process_memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "system_memory_percent": round(system_memory.percent, 2),
                "system_memory_available_gb": round(system_memory.available / 1024 / 1024 / 1024, 2),
                "cpu_percent": round(process.cpu_percent(), 2),
                "process_threads": process.num_threads(),
            },

            # Request metrics
            "requests": {
                "total": metrics_collector.request_counts["total"],
                "by_endpoint": {
                    "search": metrics_collector.request_counts["search"],
                    "insert": metrics_collector.request_counts["insert"],
                    "health": metrics_collector.request_counts["health"],
                    "stats": metrics_collector.request_counts["stats"],
                    "metrics": metrics_collector.request_counts["metrics"],
                },
                "errors": metrics_collector.request_counts["errors"],
                "error_rate_percent": round(
                    (metrics_collector.request_counts["errors"] / max(metrics_collector.request_counts["total"], 1)) * 100,
                    2
                ),
            },

            # Search performance metrics
            "search_performance": {
                "total_searches": metrics_collector.search_stats["total_searches"],
                "total_results_returned": metrics_collector.search_stats["total_results_returned"],
                "avg_results_per_search": round(
                    metrics_collector.search_stats["total_results_returned"] / max(metrics_collector.search_stats["total_searches"], 1),
                    2
                ),
                "avg_search_time_ms": round(metrics_collector.search_stats["avg_search_time"] * 1000, 2),
                "recent_search_count": len(metrics_collector.search_stats["search_times"]),
            },

            # Milvus metrics
            "milvus": milvus_metrics,

            # Rate limiting info
            "rate_limiting": {
                "enabled": True,
                "limit_per_minute": 100,
                "note": "Check X-RateLimit-* headers on responses"
            }
        }

        return response

    except Exception as e:
        # Fallback basic metrics if detailed collection fails
        metrics_collector.increment_error()
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "paper-search-api",
            "version": "1.0.0",
            "basic_metrics": {
                "uptime_seconds": round(time.time() - metrics_collector.start_time, 2),
                "total_requests": metrics_collector.request_counts["total"],
                "error": f"Detailed metrics unavailable: {str(e)[:100]}"
            }
        }

from fastapi import Response

@app.get("/prom_metrics")
async def prom_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics_collector.increment_request("metrics")

    # Build Prometheus text-formatted metrics
    lines = []

    # Total request count
    lines.append(f'paper_api_requests_total {metrics_collector.request_counts["total"]}')

    # Errors
    lines.append(f'paper_api_request_errors_total {metrics_collector.request_counts["errors"]}')

    # Requests per endpoint
    for endpoint, count in metrics_collector.request_counts.items():
        if endpoint in ["total", "errors"]:
            continue
        lines.append(
            f'paper_api_requests_by_endpoint{{endpoint="{endpoint}"}} {count}'
        )

    # Search metrics
    s = metrics_collector.search_stats
    lines.append(f'paper_api_search_total {s["total_searches"]}')
    lines.append(f'paper_api_search_results_total {s["total_results_returned"]}')
    lines.append(f'paper_api_avg_search_time_seconds {s["avg_search_time"]}')

    # System-level metrics (optional)
    process = psutil.Process(os.getpid())
    lines.append(
        f'paper_api_memory_mb {round(process.memory_info().rss / 1024 / 1024, 2)}'
    )
    lines.append(
        f'paper_api_cpu_percent {round(process.cpu_percent(), 2)}'
    )

    # Join all results
    prom_text = "\n".join(lines) + "\n"

    return Response(content=prom_text, media_type="text/plain")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    metrics_collector.increment_error()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__}
    )