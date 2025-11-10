"""
Configuration for embedding generation pipeline.
Optimized for M2 MacBook Air performance.
"""
from pathlib import Path
from typing import Dict, Any


# Base paths
BASE_DIR = Path(__file__).parent.parent
EMBED_DIR = BASE_DIR / "embed"
DATA_DIR = BASE_DIR / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
EMBEDDINGS_DIR = EMBED_DIR / "embeddings"
REPORTS_DIR = EMBED_DIR / "reports"


class EmbeddingConfig:
    """Embedding generation configuration optimized for M2."""
    
    # Model configuration
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Model output dimension
    
    # M2 Optimization
    BATCH_SIZE = 64   # Conservative for MPS stability
    NUM_WORKERS = 6   # Leave 2 cores for system (M2 has 8 cores)
    USE_MPS = True    # Use Metal Performance Shaders (Apple Silicon GPU)
    
    # Processing
    MAX_SEQ_LENGTH = 512  # Maximum tokens per text
    NORMALIZE_EMBEDDINGS = True  # For cosine similarity
    SHOW_PROGRESS = True
    
    # Memory management
    USE_MEMMAP = True  # Use memory-mapped files for large arrays
    MEMMAP_MODE = "w+"  # Write mode for memmap
    CACHE_DIR = EMBED_DIR / "model_cache"
    
    # Output format
    SAVE_FORMAT = "hybrid"  # Options: "npy", "parquet", "hybrid" (both)
    NPY_COMPRESSION = False  # .npy files don't support compression
    PARQUET_COMPRESSION = "snappy"
    
    # Validation
    VALIDATE_EMBEDDINGS = True
    CHECK_ALIGNMENT = True  # Ensure row order matches metadata
    
    # Logging
    LOG_INTERVAL = 1000  # Log progress every N records
    SAVE_CHECKPOINT_EVERY = 50000  # Save intermediate results
    
    # Quality checks
    MIN_EMBEDDING_NORM = 0.1  # Catch zero/near-zero embeddings
    MAX_EMBEDDING_NORM = 10.0  # Catch outliers
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }


# Export
__all__ = [
    "EmbeddingConfig",
    "BASE_DIR",
    "EMBED_DIR",
    "DATA_DIR",
    "CLEAN_DATA_DIR",
    "SAMPLE_DATA_DIR",
    "EMBEDDINGS_DIR",
    "REPORTS_DIR",
]
