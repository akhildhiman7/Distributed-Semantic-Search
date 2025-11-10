"""
Configuration file for data processing pipeline.
Contains all parameters, paths, and settings for reproducible data engineering.
"""
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = BASE_DIR / "arxiv-metadata-oai-snapshot.json"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
STATS_DIR = DATA_DIR / "stats"

# Processing parameters
class ProcessingConfig:
    """Data processing configuration."""
    
    # Input validation
    MIN_ABSTRACT_LENGTH = 100  # characters
    MIN_TITLE_LENGTH = 10      # characters
    MAX_ABSTRACT_LENGTH = 10000  # to filter out corrupted entries
    MAX_TITLE_LENGTH = 512
    
    # Cleaning parameters
    STRIP_HTML = True
    STRIP_LATEX = True
    NORMALIZE_WHITESPACE = True
    REMOVE_URLS = True
    
    # Deduplication
    DEDUP_HASH_CHARS = 200  # first N chars of abstract for dedup hash
    
    # Text concatenation
    TEXT_SEPARATOR = ". "  # separator between title and abstract
    
    # Partitioning
    PARTITION_SIZE_MB = 150  # target size per partition in MB
    COMPRESSION = "snappy"    # parquet compression: snappy, gzip, or None
    
    # Sampling
    SAMPLE_SIZE = 10000  # number of records for sample dataset
    SAMPLE_RANDOM_SEED = 42  # for reproducibility
    
    # Categories to include (empty list means all)
    # Focus on ML/AI/CS categories for better semantic search quality
    INCLUDE_CATEGORIES = [
        "cs.LG",  # Machine Learning
        "cs.AI",  # Artificial Intelligence
        "cs.CL",  # Computation and Language
        "cs.CV",  # Computer Vision
        "cs.IR",  # Information Retrieval
        "cs.NE",  # Neural and Evolutionary Computing
        "stat.ML",  # Machine Learning (Statistics)
    ]
    
    # Performance (optimized for M2 with 8 cores)
    BATCH_SIZE = 5000   # records per batch per core (will be multiplied by num_cores)
    LOG_INTERVAL = 50000  # log progress every N records
    USE_MULTIPROCESSING = True  # Enable parallel processing
    
    # Quality checks
    ENABLE_VALIDATION = True
    MAX_ERRORS_PER_BATCH = 100  # max parsing errors before stopping
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }


# Schema definition
ARXIV_SCHEMA = {
    "paper_id": "int64",      # Will be generated from id field
    "title": "string",
    "abstract": "string",
    "categories": "string",
    "text": "string",         # concatenated title + abstract
    "text_length": "int32",
    "has_full_data": "bool",
}

# Export settings
__all__ = [
    "ProcessingConfig",
    "BASE_DIR",
    "DATA_DIR", 
    "RAW_DATA_PATH",
    "CLEAN_DATA_DIR",
    "SAMPLE_DATA_DIR",
    "STATS_DIR",
    "ARXIV_SCHEMA",
]
