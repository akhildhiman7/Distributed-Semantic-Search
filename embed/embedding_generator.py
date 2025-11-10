"""
High-performance embedding generation pipeline optimized for M2 MacBook Air.
Uses sentence-transformers with MPS acceleration and memory-mapped arrays.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

from config import (
    EmbeddingConfig,
    CLEAN_DATA_DIR,
    SAMPLE_DATA_DIR,
    EMBEDDINGS_DIR,
    REPORTS_DIR,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(REPORTS_DIR / f'embedding_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Production-grade embedding generation with M2 optimization."""
    
    def __init__(self, config: type = EmbeddingConfig):
        self.config = config
        
        # Statistics
        self.stats = {
            "total_texts": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_time_seconds": 0,
            "texts_per_second": 0,
            "avg_text_length": 0,
            "model_name": config.MODEL_NAME,
            "embedding_dim": config.EMBEDDING_DIM,
            "batch_size": config.BATCH_SIZE,
            "device": None,
            "partitions_processed": [],
        }
        
        # Create output directories
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.device = None
        
    def _setup_device(self) -> str:
        """Setup optimal device for M2 (MPS > CPU)."""
        if self.config.USE_MPS and torch.backends.mps.is_available():
            device = "mps"
            logger.info("✓ Using Metal Performance Shaders (MPS) - Apple Silicon GPU")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("✓ Using CUDA GPU")
        else:
            device = "cpu"
            logger.info("Using CPU")
        
        return device
    
    def load_model(self):
        """Load sentence-transformer model with M2 optimization."""
        logger.info(f"Loading model: {self.config.MODEL_NAME}")
        
        # Setup device
        self.device = self._setup_device()
        
        # Load model initially on CPU to avoid MPS issues during loading
        self.model = SentenceTransformer(
            self.config.MODEL_NAME,
            cache_folder=str(self.config.CACHE_DIR),
            device="cpu",  # Load on CPU first
        )
        
        # Configure model
        self.model.max_seq_length = self.config.MAX_SEQ_LENGTH
        
        # Now move to target device if it's MPS
        if self.device == "mps":
            logger.info("Moving model to MPS device...")
            try:
                # Move model to MPS
                self.model = self.model.to(self.device)
                logger.info("✓ Model successfully moved to MPS")
            except Exception as e:
                logger.warning(f"Failed to move to MPS: {e}. Using CPU instead.")
                self.device = "cpu"
        
        # Verify embedding dimension with small test
        test_embedding = self.model.encode("test", show_progress_bar=False, device=self.device)
        actual_dim = len(test_embedding)
        
        if actual_dim != self.config.EMBEDDING_DIM:
            logger.warning(
                f"Model dimension {actual_dim} != configured {self.config.EMBEDDING_DIM}"
            )
            self.config.EMBEDDING_DIM = actual_dim
        
        self.stats["device"] = self.device
        self.stats["embedding_dim"] = actual_dim
        
        logger.info(f"✓ Model loaded: {actual_dim}-dimensional embeddings on {self.device}")
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # For MPS, process in smaller chunks to avoid memory issues
        if self.device == "mps":
            # Use smaller batches and convert to CPU after each batch
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.BATCH_SIZE,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                convert_to_tensor=False,  # Stay in numpy to avoid MPS memory issues
                normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS,
            )
        else:
            # Standard encoding for CPU/CUDA
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.BATCH_SIZE,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS,
                device=self.device,
            )
        
        return embeddings
    
    def validate_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate embedding quality.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.config.VALIDATE_EMBEDDINGS:
            return True, None
        
        # Check shape
        if embeddings.shape[0] != len(texts):
            return False, f"Shape mismatch: {embeddings.shape[0]} embeddings != {len(texts)} texts"
        
        if embeddings.shape[1] != self.config.EMBEDDING_DIM:
            return False, f"Wrong dimension: {embeddings.shape[1]} != {self.config.EMBEDDING_DIM}"
        
        # Check for NaN or Inf
        if np.any(np.isnan(embeddings)):
            return False, "NaN values in embeddings"
        
        if np.any(np.isinf(embeddings)):
            return False, "Inf values in embeddings"
        
        # Check norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        if np.any(norms < self.config.MIN_EMBEDDING_NORM):
            return False, f"Embedding norms too small: min={norms.min():.4f}"
        
        if np.any(norms > self.config.MAX_EMBEDDING_NORM):
            return False, f"Embedding norms too large: max={norms.max():.4f}"
        
        return True, None
    
    def process_partition(
        self,
        partition_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Process a single partition file.
        
        Args:
            partition_path: Path to input parquet file
            output_dir: Directory for output files
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing: {partition_path.name}")
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read partition
        df = pd.read_parquet(partition_path)
        texts = df['text'].tolist()
        
        logger.info(f"  Loaded {len(texts):,} texts")
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(
            texts,
            show_progress=self.config.SHOW_PROGRESS,
        )
        
        # Validate
        is_valid, error_msg = self.validate_embeddings(embeddings, texts)
        if not is_valid:
            raise ValueError(f"Embedding validation failed: {error_msg}")
        
        # Save outputs
        partition_name = partition_path.stem  # e.g., "part-0000"
        
        # Save embeddings
        if self.config.SAVE_FORMAT in ["npy", "hybrid"]:
            npy_path = output_dir / f"{partition_name}_embeddings.npy"
            np.save(npy_path, embeddings)
            logger.info(f"  ✓ Saved embeddings: {npy_path.name}")
        
        # Save metadata + embeddings in parquet
        if self.config.SAVE_FORMAT in ["parquet", "hybrid"]:
            # Create output dataframe
            df_out = df.copy()
            
            # Add embedding columns (can also save as list/array column)
            # For Milvus integration, we'll keep embeddings separate
            # but save alignment info
            df_out['embedding_index'] = range(len(df))
            df_out['embedding_norm'] = np.linalg.norm(embeddings, axis=1)
            
            parquet_path = output_dir / f"{partition_name}_metadata.parquet"
            df_out.to_parquet(
                parquet_path,
                compression=self.config.PARQUET_COMPRESSION,
                index=False,
            )
            logger.info(f"  ✓ Saved metadata: {parquet_path.name}")
        
        # Statistics
        elapsed = time.time() - start_time
        texts_per_sec = len(texts) / elapsed if elapsed > 0 else 0
        
        partition_stats = {
            "partition": partition_path.name,
            "num_texts": len(texts),
            "embedding_shape": embeddings.shape,
            "elapsed_seconds": elapsed,
            "texts_per_second": texts_per_sec,
            "avg_text_length": df['text_length'].mean(),
        }
        
        logger.info(
            f"  ✓ Complete: {len(texts):,} embeddings in {elapsed:.1f}s "
            f"({texts_per_sec:.1f} texts/sec)"
        )
        
        return partition_stats
    
    def process_all_partitions(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "part-*.parquet",
    ):
        """
        Process all partition files.
        
        Args:
            input_dir: Directory with input parquet files
            output_dir: Directory for output embeddings
            pattern: File pattern to match
        """
        # Get all partition files
        partition_files = sorted(input_dir.glob(pattern))
        
        if not partition_files:
            logger.error(f"No files matching {pattern} in {input_dir}")
            return
        
        logger.info(f"Found {len(partition_files)} partitions to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each partition
        total_start = time.time()
        
        for i, partition_path in enumerate(partition_files, 1):
            logger.info(f"\n[{i}/{len(partition_files)}] " + "="*50)
            
            try:
                partition_stats = self.process_partition(partition_path, output_dir)
                
                # Update global stats
                self.stats["total_texts"] += partition_stats["num_texts"]
                self.stats["successful_embeddings"] += partition_stats["num_texts"]
                self.stats["avg_text_length"] += partition_stats["avg_text_length"]
                self.stats["partitions_processed"].append(partition_stats)
                
            except Exception as e:
                logger.error(f"Error processing {partition_path.name}: {e}")
                self.stats["failed_embeddings"] += 1
                continue
        
        # Finalize stats
        total_elapsed = time.time() - total_start
        self.stats["total_time_seconds"] = total_elapsed
        
        if self.stats["total_texts"] > 0:
            self.stats["texts_per_second"] = self.stats["total_texts"] / total_elapsed
            self.stats["avg_text_length"] /= len(partition_files)
        
        logger.info("\n" + "="*70)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total texts:        {self.stats['total_texts']:,}")
        logger.info(f"Successful:         {self.stats['successful_embeddings']:,}")
        logger.info(f"Failed:             {self.stats['failed_embeddings']:,}")
        logger.info(f"Total time:         {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        logger.info(f"Throughput:         {self.stats['texts_per_second']:.1f} texts/sec")
        logger.info(f"Avg text length:    {self.stats['avg_text_length']:.1f} chars")
        logger.info(f"Device:             {self.stats['device']}")
        logger.info(f"Embedding dim:      {self.stats['embedding_dim']}")
        logger.info("="*70 + "\n")
    
    def save_report(self):
        """Save processing report."""
        report_path = REPORTS_DIR / f"speed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")


def main():
    """Main embedding generation pipeline."""
    logger.info("="*70)
    logger.info("EMBEDDING GENERATION PIPELINE - M2 OPTIMIZED")
    logger.info("="*70)
    logger.info(f"Configuration: {EmbeddingConfig.to_dict()}\n")
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    
    # Load model
    pipeline.load_model()
    
    # Check for sample data first (for quick testing)
    sample_file = SAMPLE_DATA_DIR / "sample.parquet"
    if sample_file.exists():
        logger.info("\n" + "="*70)
        logger.info("PROCESSING SAMPLE DATASET (10k records)")
        logger.info("="*70)
        
        sample_output = EMBEDDINGS_DIR / "sample"
        partition_stats = pipeline.process_partition(sample_file, sample_output)
        
        logger.info("\n✓ Sample processing complete!")
        logger.info(f"  Output: {sample_output}")
        logger.info(f"  Throughput: {partition_stats['texts_per_second']:.1f} texts/sec")
    
    # Process full dataset
    if CLEAN_DATA_DIR.exists() and list(CLEAN_DATA_DIR.glob("part-*.parquet")):
        logger.info("\n" + "="*70)
        logger.info("PROCESSING FULL DATASET")
        logger.info("="*70)
        
        full_output = EMBEDDINGS_DIR / "full"
        pipeline.process_all_partitions(CLEAN_DATA_DIR, full_output)
    else:
        logger.warning("No full dataset found in data/clean/")
    
    # Save report
    pipeline.save_report()
    
    logger.info("\n✓ PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()
