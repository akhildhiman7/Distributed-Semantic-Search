from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import os
import psutil
from typing import List

from prometheus_client import Counter, Gauge, generate_latest, CollectorRegistry

from api.middleware import rate_limit_middleware
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

# ----------------------------
# Prometheus metrics
# ----------------------------
registry = CollectorRegistry()
REQUEST_COUNTER = Counter(
    "api_requests_total", "Total API requests", ["endpoint"], registry=registry
)
ERROR_COUNTER = Counter(
    "api_errors_total", "Total API errors", ["endpoint"], registry=registry
)
SEARCH_TIME = Gauge(
    "api_search_time_seconds", "Time taken for search requests", registry=registry
)
SEARCH_RESULTS = Gauge(
    "api_search_results_total", "Number of results returned per search", registry=registry
)

# ----------------------------
# JSON metrics collector
# ----------------------------
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
        REQUEST_COUNTER.labels(endpoint=endpoint).inc()
    
    def increment_error(self, endpoint: str = "unknown"):
        self.request_counts["errors"] += 1
        ERROR_COUNTER.labels(endpoint=endpoint).inc()
    
    def record_search(self, num_results: int, search_time: float):
        self.search_stats["total_searches"] += 1
        self.search_stats["total_results_returned"] += num_results
        self.search_stats["search_times"].append(search_time)
        if len(self.search_stats["search_times"]) > 100:
            self.search_stats["search_times"] = self.search_stats["search_times"][-100:]
        if self.search_stats["search_times"]:
            self.search_stats["avg_search_time"] = sum(self.search_stats["search_times"]) / len(self.search_stats["search_times"])
        SEARCH_TIME.set(search_time)
        SEARCH_RESULTS.set(num_results)

metrics_collector = MetricsCollector()

# ----------------------------
# Helpers & startup
# ----------------------------
def get_milvus_config():
    return MilvusConfig()

@app.on_event("startup")
async def startup_event():
    global milvus_adapter, embedder
    try:
        config = get_milvus_config()
        milvus_adapter = MilvusAdapter(config)
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

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
async def root():
    metrics_collector.increment_request("root")
    return {
        "message": "Paper Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search",
            "insert": "POST /insert",
            "health": "GET /health",
            "stats": "GET /stats",
            "metrics": "GET /metrics",
            "prometheus_metrics": "GET /metrics_prometheus"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
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
        metrics_collector.increment_error("health")
        return HealthResponse(
            status="unhealthy",
            milvus_connected=False,
            collection_loaded=False,
            num_papers=0,
            timestamp=datetime.now()
        )

@app.get("/stats")
async def stats():
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
        metrics_collector.increment_error("stats")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[PaperResponse])
async def search(request: SearchRequest):
    metrics_collector.increment_request("search")
    if milvus_adapter is None:
        metrics_collector.increment_error("search")
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    start_time = time.time()
    try:
        if embedder is None:
            metrics_collector.increment_error("search")
            raise HTTPException(
                status_code=501, 
                detail="Embedder not available. Install sentence-transformers or provide pre-computed vector."
            )
        query_embedding = embedder.encode(request.query).tolist()
        results = milvus_adapter.search(
            vector=query_embedding,
            top_k=request.top_k,
            categories=request.categories
        )
        if request.min_score > 0:
            results = [r for r in results if r["score"] >= request.min_score]
        response = [PaperResponse(**paper) for paper in results]
        elapsed = time.time() - start_time
        metrics_collector.record_search(len(response), elapsed)
        print(f"🔍 Search: '{request.query[:50]}...' - {len(response)} results in {elapsed:.3f}s")
        return response
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.increment_error("search")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/insert", response_model=InsertResponse)
async def insert(request: InsertRequest):
    metrics_collector.increment_request("insert")
    if milvus_adapter is None:
        metrics_collector.increment_error("insert")
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    try:
        expected_dim = get_milvus_config().dim
        if len(request.vector) != expected_dim:
            metrics_collector.increment_error("insert")
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch. Expected {expected_dim}, got {len(request.vector)}"
            )
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
        metrics_collector.increment_error("insert")
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Return detailed JSON metrics (for dashboards)."""
    metrics_collector.increment_request("metrics")
    try:
        process = psutil.Process(os.getpid())
        system_memory = psutil.virtual_memory()
        uptime_seconds = time.time() - metrics_collector.start_time
        uptime_hours = uptime_seconds / 3600
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
        return {
            "timestamp": datetime.now().isoformat(),
            "service": "paper-search-api",
            "version": "1.0.0",
            "system": {
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_hours": round(uptime_hours, 2),
                "process_memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "system_memory_percent": round(system_memory.percent, 2),
                "system_memory_available_gb": round(system_memory.available / 1024 / 1024 / 1024, 2),
                "cpu_percent": round(process.cpu_percent(), 2),
                "process_threads": process.num_threads(),
            },
            "requests": metrics_collector.request_counts,
            "search_performance": metrics_collector.search_stats,
            "milvus": milvus_metrics
        }
    except Exception as e:
        metrics_collector.increment_error("metrics")
        return {"error": str(e)}

@app.get("/metrics_prometheus")
async def metrics_prometheus():
    """Prometheus-compatible metrics endpoint"""
    metrics_collector.increment_request("metrics_prometheus")
    return Response(generate_latest(registry), media_type="text/plain")

# ----------------------------
# Global exception handler
# ----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    metrics_collector.increment_error("global")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": exc.__class__.__name__}
    )
