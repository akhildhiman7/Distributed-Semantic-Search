"""
Configuration for FastAPI service.
"""
import os
from typing import List

# Milvus Connection
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = "arxiv_papers"

# Model Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

# Search Parameters
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
MIN_TOP_K = 1
DEFAULT_MIN_SCORE = 0.0

# Milvus Search Parameters
SEARCH_PARAMS = {
    "metric_type": "IP",  # Inner Product
    "params": {"nprobe": 16}
}

# API Configuration
API_TITLE = "Distributed Semantic Search API"
API_DESCRIPTION = """
**Semantic search API for arXiv papers** powered by Milvus vector database.

Search through 510K+ research papers using natural language queries.

## Features
- 🔍 Semantic search with sentence embeddings
- ⚡ Fast vector similarity search (<250ms)
- 📊 510,203 indexed research papers
- 🎯 Category filtering support
- 📈 Performance metrics and health checks

## Model
- **Encoder**: sentence-transformers/all-MiniLM-L6-v2
- **Embedding Dimension**: 384
- **Similarity Metric**: Inner Product (cosine similarity)

## Dataset
- **Source**: arXiv metadata
- **Categories**: cs.LG, cs.AI, cs.CL, cs.CV, cs.IR, cs.NE, stat.ML
- **Size**: 510,203 papers
"""
API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "Member 4 - API Developer",
    "email": "member4@example.com"
}

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Performance
REQUEST_TIMEOUT = 30  # seconds
MAX_CONCURRENT_REQUESTS = 100

# Categories for filtering
VALID_CATEGORIES = [
    "cs.LG", "cs.AI", "cs.CL", "cs.CV",
    "cs.IR", "cs.NE", "stat.ML"
]
