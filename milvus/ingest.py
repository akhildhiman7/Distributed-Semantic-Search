"""
Milvus Data Ingestion Pipeline
Ingest embeddings and metadata from parquet/numpy files into Milvus collection
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pymilvus import connections, Collection, utility
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBEDDINGS_DIR,
    BATCH_SIZE, CHECKPOINT_INTERVAL, CHECKPOINT_FILE, LOG_LEVEL, LOG_FILE
)
from schema import create_collection_schema, get_collection_properties

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MilvusIngestor:
    """
    Handles ingestion of embeddings and metadata into Milvus
    """
    
    def __init__(self):
        self.collection = None
        self.checkpoint = self._load_checkpoint()
        
    def connect(self):
        """Connect to Milvus server"""
        logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        logger.info("✓ Connected to Milvus")
        
    def create_collection(self, drop_existing=False):
        """Create or load collection"""
        if drop_existing and utility.has_collection(COLLECTION_NAME):
            logger.warning(f"Dropping existing collection: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME)
        
        if not utility.has_collection(COLLECTION_NAME):
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            schema = create_collection_schema()
            properties = get_collection_properties()
            
            self.collection = Collection(
                name=COLLECTION_NAME,
                schema=schema,
                **properties
            )
            logger.info("✓ Collection created")
        else:
            logger.info(f"Loading existing collection: {COLLECTION_NAME}")
            self.collection = Collection(COLLECTION_NAME)
            logger.info("✓ Collection loaded")
    
    def _load_checkpoint(self) -> Dict:
        """Load ingestion checkpoint"""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        return {"completed_partitions": [], "total_inserted": 0}
    
    def _save_checkpoint(self):
        """Save ingestion checkpoint"""
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def load_partition_data(self, partition_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load embeddings and metadata for a partition"""
        embeddings_path = f"{EMBEDDINGS_DIR}/{partition_name}_embeddings.npy"
        metadata_path = f"{EMBEDDINGS_DIR}/{partition_name}_metadata.parquet"
        
        logger.info(f"Loading: {embeddings_path}")
        embeddings = np.load(embeddings_path, mmap_mode='r')  # Memory-mapped for efficiency
        
        logger.info(f"Loading: {metadata_path}")
        metadata = pd.read_parquet(metadata_path)
        
        # Validate alignment
        assert len(embeddings) == len(metadata), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata rows"
        
        logger.info(f"✓ Loaded {len(embeddings):,} records from {partition_name}")
        return embeddings, metadata
    
    def prepare_batch_data(self, embeddings: np.ndarray, metadata: pd.DataFrame, 
                          start_idx: int, end_idx: int) -> List[List]:
        """
        Prepare batch data in Milvus insert format
        Returns: [paper_ids, embeddings, titles, abstracts, categories, text_lengths, has_full_data]
        """
        batch_meta = metadata.iloc[start_idx:end_idx]
        batch_embed = embeddings[start_idx:end_idx]
        
        # Convert to lists for Milvus insertion
        # Ensure paper_id is string type
        data = [
            batch_meta['paper_id'].astype(str).tolist(),
            batch_embed.tolist(),
            batch_meta['title'].tolist(),
            batch_meta['abstract'].tolist(),
            batch_meta['categories'].tolist(),
            batch_meta['text_length'].tolist(),
            batch_meta['has_full_data'].tolist()
        ]
        
        return data
    
    def ingest_partition(self, partition_name: str):
        """Ingest a single partition into Milvus"""
        if partition_name in self.checkpoint["completed_partitions"]:
            logger.info(f"⏭  Skipping {partition_name} (already completed)")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingesting: {partition_name}")
        logger.info(f"{'='*60}")
        
        # Load data
        embeddings, metadata = self.load_partition_data(partition_name)
        total_records = len(embeddings)
        
        # Insert in batches
        num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
        inserted = 0
        
        with tqdm(total=total_records, desc=f"Inserting {partition_name}") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, total_records)
                
                # Prepare batch
                batch_data = self.prepare_batch_data(embeddings, metadata, start_idx, end_idx)
                
                # Insert batch
                try:
                    self.collection.insert(batch_data)
                    batch_size = end_idx - start_idx
                    inserted += batch_size
                    pbar.update(batch_size)
                    
                    # Save checkpoint periodically
                    if inserted % CHECKPOINT_INTERVAL == 0:
                        self.checkpoint["total_inserted"] += batch_size
                        self._save_checkpoint()
                
                except Exception as e:
                    logger.error(f"Error inserting batch {batch_idx}: {e}")
                    raise
        
        # Mark partition as complete
        self.checkpoint["completed_partitions"].append(partition_name)
        self.checkpoint["total_inserted"] += inserted
        self._save_checkpoint()
        
        logger.info(f"✓ Inserted {inserted:,} records from {partition_name}")
        logger.info(f"✓ Total inserted so far: {self.checkpoint['total_inserted']:,}")
    
    def ingest_all_partitions(self):
        """Ingest all partition files"""
        # Find all partition files
        partition_files = sorted([
            f.replace("_embeddings.npy", "") 
            for f in os.listdir(EMBEDDINGS_DIR) 
            if f.endswith("_embeddings.npy")
        ])
        
        logger.info(f"\nFound {len(partition_files)} partitions to ingest")
        logger.info(f"Partitions: {partition_files}\n")
        
        start_time = time.time()
        
        for partition_name in partition_files:
            self.ingest_partition(partition_name)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ INGESTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total records inserted: {self.checkpoint['total_inserted']:,}")
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Average rate: {self.checkpoint['total_inserted']/elapsed:.0f} records/sec")
    
    def flush_and_compact(self):
        """Flush data to disk and compact"""
        logger.info("\nFlushing collection to disk...")
        self.collection.flush()
        logger.info("✓ Flush complete")
        
        # Compact to optimize storage
        logger.info("Compacting collection...")
        self.collection.compact()
        logger.info("✓ Compaction complete")
    
    def create_index(self, index_type="IVF_FLAT"):
        """Create index on vector field"""
        from config import INDEX_PARAMS
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Building {index_type} index...")
        logger.info(f"{'='*60}")
        
        index_params = INDEX_PARAMS[index_type]
        
        start_time = time.time()
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Index built in {elapsed/60:.2f} minutes")
    
    def load_collection(self):
        """Load collection into memory for querying"""
        logger.info("\nLoading collection into memory...")
        self.collection.load()
        logger.info("✓ Collection loaded and ready for queries")
    
    def get_collection_stats(self):
        """Get collection statistics"""
        stats = {
            "total_entities": self.collection.num_entities,
            "has_index": len(self.collection.indexes) > 0,
            "loaded": utility.load_state(COLLECTION_NAME)
        }
        return stats
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        logger.info("✓ Disconnected from Milvus")


def main(drop_existing=False, index_type="IVF_FLAT"):
    """
    Main ingestion pipeline
    
    Args:
        drop_existing: Whether to drop existing collection
        index_type: Type of index to build (IVF_FLAT or HNSW)
    """
    ingestor = MilvusIngestor()
    
    try:
        # 1. Connect to Milvus
        ingestor.connect()
        
        # 2. Create/load collection
        ingestor.create_collection(drop_existing=drop_existing)
        
        # 3. Ingest all partitions
        ingestor.ingest_all_partitions()
        
        # 4. Flush and compact
        ingestor.flush_and_compact()
        
        # 5. Build index
        ingestor.create_index(index_type=index_type)
        
        # 6. Load collection
        ingestor.load_collection()
        
        # 7. Print stats
        stats = ingestor.get_collection_stats()
        logger.info(f"\n{'='*60}")
        logger.info("COLLECTION STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total entities: {stats['total_entities']:,}")
        logger.info(f"Has index: {stats['has_index']}")
        logger.info(f"Loaded: {stats['loaded']}")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise
    finally:
        ingestor.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest embeddings into Milvus")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection")
    parser.add_argument("--index", default="IVF_FLAT", choices=["IVF_FLAT", "HNSW"],
                       help="Index type to build")
    
    args = parser.parse_args()
    
    main(drop_existing=args.drop, index_type=args.index)
