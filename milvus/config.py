"""
Milvus Configuration
Centralized configuration for Milvus cluster connection and collection settings
"""

import os

# Milvus Connection Settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

# Collection Settings
COLLECTION_NAME = "arxiv_papers"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 embedding dimension
METRIC_TYPE = "IP"  # Inner Product (for normalized vectors), alternatives: "L2", "COSINE"

# Index Settings
INDEX_TYPE = "IVF_FLAT"  # Faster build time, good balance
# Alternative: "HNSW" for better query performance but slower build
INDEX_PARAMS = {
    "IVF_FLAT": {
        "metric_type": METRIC_TYPE,
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 128  # Number of clusters (lower = faster build, slightly slower search)
        }
    },
    "HNSW": {
        "metric_type": METRIC_TYPE,
        "index_type": "HNSW",
        "params": {
            "M": 16,  # Max connections per node
            "efConstruction": 200  # Build time quality
        }
    }
}

# Search Parameters
SEARCH_PARAMS = {
    "IVF_FLAT": {
        "metric_type": METRIC_TYPE,
        "params": {
            "nprobe": 16  # Number of clusters to search
        }
    },
    "HNSW": {
        "metric_type": METRIC_TYPE,
        "params": {
            "ef": 64  # Search quality parameter
        }
    }
}

# Ingestion Settings
BATCH_SIZE = 1000  # Vectors per insert batch
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every N records
CHECKPOINT_FILE = "milvus/checkpoints/ingestion_checkpoint.json"

# Data Paths
EMBEDDINGS_DIR = "embed/embeddings/full"
SAMPLE_EMBEDDINGS_DIR = "embed/embeddings/sample"

# Sharding and Replication (for distributed setup)
NUM_SHARDS = 2  # Number of shards for distributing data
NUM_REPLICAS = 1  # Number of replicas for fault tolerance (set to 2-3 in production)

# Performance Settings
CONSISTENCY_LEVEL = "Eventually"  # "Strong", "Eventually", "Bounded", "Session"
# Eventually = fastest, Strong = most consistent

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "milvus/logs/milvus_ingestion.log"
