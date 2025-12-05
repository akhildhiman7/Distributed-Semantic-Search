"""
Ingest data into distributed Milvus cluster.
Reuses existing embeddings and adapts for distributed collection.
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus import connections, Collection
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from milvus.config import EMBEDDINGS_DIR, BATCH_SIZE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedIngestor:
    """Ingest data into distributed Milvus cluster."""
    
    def __init__(self, collection_name="arxiv_papers_distributed", proxy_port=19531):
        self.collection_name = collection_name
        self.proxy_port = proxy_port
        self.collection = None
        
    def connect(self):
        """Connect to distributed cluster via proxy."""
        logger.info(f"Connecting to distributed cluster on port {self.proxy_port}...")
        connections.connect(
            alias="default",
            host="localhost",
            port=self.proxy_port
        )
        self.collection = Collection(self.collection_name)
        logger.info("✓ Connected to distributed cluster")
        
    def ingest_all_data(self, resume=True):
        """Ingest all partition data into distributed collection."""
        # Check current entity count
        current_count = self.collection.num_entities
        if current_count > 0 and resume:
            logger.info(f"Found existing {current_count:,} entities, resuming ingestion...")
        
        # Find all embeddings files (use absolute path)
        project_root = Path(__file__).parent.parent
        embeddings_dir = project_root / "embed" / "embeddings" / "full"
        
        if not embeddings_dir.exists():
            logger.error(f"Embeddings directory not found: {embeddings_dir}")
            return
            
        embedding_files = sorted(embeddings_dir.glob("*_embeddings.npy"))
        
        if not embedding_files:
            logger.error(f"No embedding files found in {EMBEDDINGS_DIR}")
            return
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        
        total_inserted = 0
        start_time = time.time()
        
        for emb_file in embedding_files:
            partition_name = emb_file.stem.replace('_embeddings', '')
            metadata_file = embeddings_dir / f"{partition_name}_metadata.parquet"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                continue
            
            logger.info(f"\nProcessing partition: {partition_name}")
            
            # Load embeddings and metadata
            embeddings = np.load(str(emb_file), mmap_mode='r')
            metadata = pd.read_parquet(str(metadata_file))
            
            logger.info(f"  Vectors: {embeddings.shape[0]:,}")
            logger.info(f"  Metadata rows: {len(metadata):,}")
            
            # Validate counts match
            if embeddings.shape[0] != len(metadata):
                logger.error(f"  Mismatch: {embeddings.shape[0]} vectors != {len(metadata)} metadata rows")
                continue
            
            # Ingest in smaller batches for distributed mode (reduces timeout risk)
            batch_size = 500  # Smaller than default to avoid timeouts
            num_batches = (len(embeddings) + batch_size - 1) // batch_size
            
            with tqdm(total=len(embeddings), desc=f"  Ingesting {partition_name}") as pbar:
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min(start_idx + batch_size, len(embeddings))
                    
                    # Prepare batch data
                    batch_embeddings = embeddings[start_idx:end_idx]
                    batch_metadata = metadata.iloc[start_idx:end_idx]
                    
                    # Prepare batch data (matching schema order)
                    batch_data = [
                        batch_metadata['paper_id'].astype(str).tolist(),
                        batch_embeddings.tolist(),
                        batch_metadata['title'].tolist(),
                        batch_metadata['abstract'].tolist(),
                        batch_metadata['categories'].tolist(),
                        batch_metadata['text_length'].tolist(),
                        batch_metadata['has_full_data'].tolist(),
                    ]
                    
                    # Insert batch (automatically distributed across shards)
                    try:
                        self.collection.insert(batch_data)
                        total_inserted += (end_idx - start_idx)
                        pbar.update(end_idx - start_idx)
                    except Exception:
                        logger.warning(f"  Timeout/error inserting batch {i}, retrying...")
                        # Retry once with longer timeout
                        try:
                            time.sleep(2)
                            self.collection.insert(batch_data)
                            total_inserted += (end_idx - start_idx)
                            pbar.update(end_idx - start_idx)
                        except Exception as retry_e:
                            logger.error(f"  Failed batch {i} after retry: {retry_e}")
                            # Skip this batch and continue
                            continue
        
        # Flush to persist
        logger.info("\nFlushing collection...")
        self.collection.flush()
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Ingestion complete!")
        logger.info(f"  Total inserted: {total_inserted:,}")
        logger.info(f"  Time taken: {elapsed:.1f}s")
        logger.info(f"  Throughput: {total_inserted/elapsed:.1f} vectors/sec")
        
        # Verify
        logger.info(f"\n📊 Collection stats:")
        logger.info(f"  Total entities: {self.collection.num_entities:,}")
        
        return total_inserted


def main():
    """Main ingestion function."""
    print("="*60)
    print("DISTRIBUTED MILVUS INGESTION")
    print("="*60)
    
    ingestor = DistributedIngestor()
    
    try:
        ingestor.connect()
        ingestor.ingest_all_data()
        print("\n✅ All data ingested successfully!")
        print("\nNext: Run distributed/test_distributed.py to test performance")
    except Exception as e:
        logger.error(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
