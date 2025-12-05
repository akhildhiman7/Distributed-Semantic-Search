# api/dependencies.py
import os
from typing import List, Optional
from functools import lru_cache

from config import MilvusConfig

@lru_cache(maxsize=1)
def get_milvus_config():
    """Get Milvus configuration"""
    return MilvusConfig()

def normalize_categories(categories: Optional[List[str]]) -> str:
    """Convert categories list to string format for Milvus"""
    if not categories:
        return ""
    return ",".join(sorted(map(str, categories)))