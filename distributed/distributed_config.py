"""
Configuration for distributed Milvus cluster with sharding and replication.

This module configures:
1. Collection sharding across multiple data nodes
2. Segment replication for fault tolerance
3. Query node replicas for parallel search
"""

# ============================================
# Distributed Collection Schema
# ============================================

COLLECTION_CONFIG = {
    "name": "arxiv_papers_distributed",
    "description": "ArXiv papers with distributed sharding and replication",
    
    # Sharding configuration
    "shards_num": 4,  # Split collection across 4 shards (distributed across data nodes)
    
    # Replication configuration
    "consistency_level": "Strong",  # Strong, Bounded, Session, Eventually
    
    # Collection properties
    "properties": {
        "collection.ttl.seconds": 0,  # No TTL, keep data indefinitely
    }
}

# ============================================
# Index Configuration for Distributed Setup
# ============================================

INDEX_CONFIG = {
    # HNSW is better for distributed search (faster than IVF_FLAT)
    "index_type": "HNSW",
    "metric_type": "IP",  # Inner Product (cosine similarity)
    "params": {
        "M": 16,              # Number of bi-directional links (4-64, default 16)
        "efConstruction": 200  # Size of dynamic candidate list (100-500, default 200)
    }
}

# Alternative: IVF_FLAT for memory-constrained environments
INDEX_CONFIG_IVF = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {
        "nlist": 256  # More clusters for better distribution
    }
}

# ============================================
# Search Configuration
# ============================================

SEARCH_PARAMS_HNSW = {
    "metric_type": "IP",
    "params": {
        "ef": 64  # Size of dynamic candidate list during search (top_k to 4096)
    }
}

SEARCH_PARAMS_IVF = {
    "metric_type": "IP",
    "params": {
        "nprobe": 32  # Number of clusters to search
    }
}

# ============================================
# Replica Configuration
# ============================================

REPLICA_CONFIG = {
    # Number of replicas for the collection
    # With 3 query nodes, we can have 3 replicas
    "replica_number": 3,
    
    # Resource groups (optional, for advanced deployment)
    "resource_groups": None
}

# ============================================
# Load Balancing Configuration
# ============================================

LOAD_BALANCE_CONFIG = {
    # Load balance policy for query nodes
    "policy": "RoundRobin",  # RoundRobin, LeastLoad, Random
    
    # Query node selection
    "replica_selection": "Random"  # Random, RoundRobin, Weighted
}

# ============================================
# Connection Configuration
# ============================================

# Multiple proxy connections for client-side load balancing
MILVUS_PROXIES = [
    {"host": "localhost", "port": 19531},  # proxy-1
    {"host": "localhost", "port": 19532},  # proxy-2
]

# Connection pool settings
CONNECTION_POOL = {
    "pool_size": 10,
    "max_idle_time": 60,  # seconds
    "max_lifetime": 3600,  # seconds
}

# ============================================
# Compaction Configuration
# ============================================

COMPACTION_CONFIG = {
    # Auto compaction for better query performance
    "auto_compaction": True,
    
    # Trigger compaction when segment count exceeds threshold
    "max_segment_count": 1000,
    
    # Minimum size for compaction (bytes)
    "min_segment_size": 1024 * 1024 * 100,  # 100MB
}

# ============================================
# Resource Allocation
# ============================================

RESOURCE_LIMITS = {
    "query_node": {
        "cpu": 2,
        "memory_gb": 4,
    },
    "data_node": {
        "cpu": 2,
        "memory_gb": 4,
    },
    "index_node": {
        "cpu": 2,
        "memory_gb": 4,
    }
}

# ============================================
# Monitoring Configuration
# ============================================

MONITORING_CONFIG = {
    "enable_metrics": True,
    "metrics_port": 9091,
    
    # Alert thresholds
    "alerts": {
        "query_latency_p99_ms": 1000,
        "memory_usage_percent": 80,
        "disk_usage_percent": 85,
        "error_rate_percent": 5,
    }
}
