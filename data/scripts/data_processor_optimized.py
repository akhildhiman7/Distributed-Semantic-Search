"""
High-performance, multiprocessing-optimized data processing pipeline for M2.
Leverages all CPU cores for maximum throughput.
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import sys
import multiprocessing as mp
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    ProcessingConfig,
    RAW_DATA_PATH,
    CLEAN_DATA_DIR,
    SAMPLE_DATA_DIR,
    STATS_DIR,
)
from text_cleaner import TextCleaner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(STATS_DIR / f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def process_record_worker(record_line: str, config_dict: dict, seen_hashes_chunk: set) -> Optional[Dict[str, Any]]:
    """
    Worker function to process a single record (designed for multiprocessing).
    
    Args:
        record_line: JSON line from file
        config_dict: Configuration dictionary
        seen_hashes_chunk: Set of hashes seen in this chunk (for dedup)
        
    Returns:
        Processed record or None
    """
    try:
        # Parse JSON
        record = json.loads(record_line.strip())
        
        # Check required fields
        if not record.get("title") or not record.get("abstract"):
            return None
        
        # Check categories filter
        if config_dict['INCLUDE_CATEGORIES']:
            categories = record.get("categories", "")
            has_category = any(cat in categories for cat in config_dict['INCLUDE_CATEGORIES'])
            if not has_category:
                return None
        
        # Extract fields
        arxiv_id = record.get("id", "")
        title_raw = record.get("title", "")
        abstract_raw = record.get("abstract", "")
        categories = record.get("categories", "")
        
        # Clean text
        text_cleaner = TextCleaner()
        title_clean = text_cleaner.clean_text(
            title_raw,
            strip_latex=config_dict['STRIP_LATEX'],
            strip_html=config_dict['STRIP_HTML'],
            remove_urls=config_dict['REMOVE_URLS'],
        )
        
        abstract_clean = text_cleaner.clean_text(
            abstract_raw,
            strip_latex=config_dict['STRIP_LATEX'],
            strip_html=config_dict['STRIP_HTML'],
            remove_urls=config_dict['REMOVE_URLS'],
        )
        
        # Validate length
        if len(title_clean) < config_dict['MIN_TITLE_LENGTH']:
            return None
        
        if len(abstract_clean) < config_dict['MIN_ABSTRACT_LENGTH']:
            return None
        
        if len(abstract_clean) > config_dict['MAX_ABSTRACT_LENGTH']:
            return None
        
        # Check for duplicates
        title_norm = title_clean.lower().strip()
        abstract_prefix = abstract_clean.lower().strip()[:config_dict['DEDUP_HASH_CHARS']]
        combined = f"{title_norm}||{abstract_prefix}"
        dedup_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()
        
        if dedup_hash in seen_hashes_chunk:
            return None
        
        seen_hashes_chunk.add(dedup_hash)
        
        # Create combined text
        text = text_cleaner.create_search_text(
            title_clean,
            abstract_clean,
            separator=config_dict['TEXT_SEPARATOR']
        )
        
        # Validate final text
        is_valid, _ = text_cleaner.validate_text(
            text,
            min_length=config_dict['MIN_ABSTRACT_LENGTH'],
            max_length=config_dict['MAX_ABSTRACT_LENGTH'] + config_dict['MAX_TITLE_LENGTH']
        )
        
        if not is_valid:
            return None
        
        # Extract paper ID
        paper_id = extract_paper_id(arxiv_id)
        
        # Return processed record
        return {
            "paper_id": paper_id,
            "title": title_clean,
            "abstract": abstract_clean,
            "categories": categories,
            "text": text,
            "text_length": len(text),
            "has_full_data": True,
            "dedup_hash": dedup_hash,
        }
    
    except Exception as e:
        return None


def extract_paper_id(arxiv_id: str) -> int:
    """Convert arXiv ID to integer paper_id."""
    try:
        arxiv_id = arxiv_id.replace("arXiv:", "")
        parts = arxiv_id.split(".")
        if len(parts) != 2:
            return hash(arxiv_id) % (10**10)
        
        yymm = parts[0].replace("/", "")
        nnnnn = parts[1]
        
        if len(yymm) <= 4 and yymm.isdigit():
            paper_id = int(yymm + nnnnn.zfill(5))
        else:
            paper_id = hash(arxiv_id) % (10**10)
        
        return paper_id
    except:
        return hash(arxiv_id) % (10**10)


def process_batch_parallel(lines: List[str], config_dict: dict, num_workers: int = None) -> List[Dict[str, Any]]:
    """
    Process a batch of lines in parallel using multiprocessing.
    
    Args:
        lines: List of JSON lines to process
        config_dict: Configuration dictionary
        num_workers: Number of worker processes (None = use all cores)
        
    Returns:
        List of processed records
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Create a shared set for deduplication within this batch
    seen_hashes = set()
    
    # Process in parallel
    worker_func = partial(process_record_worker, config_dict=config_dict, seen_hashes_chunk=seen_hashes)
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_func, lines, chunksize=max(1, len(lines) // (num_workers * 4)))
    
    # Filter out None results
    processed_records = [r for r in results if r is not None]
    
    return processed_records


class OptimizedArxivDataProcessor:
    """M2-optimized, multiprocessing data processor."""
    
    def __init__(self, config: ProcessingConfig = ProcessingConfig, num_workers: int = None):
        self.config = config
        self.num_workers = num_workers or mp.cpu_count()
        
        logger.info(f"Initializing processor with {self.num_workers} worker processes")
        
        # Statistics tracking
        self.stats = {
            "total_records": 0,
            "processed_records": 0,
            "skipped_records": 0,
            "duplicate_records": 0,
            "error_records": 0,
            "category_counts": defaultdict(int),
            "total_text_bytes": 0,
            "avg_title_length": 0,
            "avg_abstract_length": 0,
            "avg_text_length": 0,
        }
        
        # Global deduplication tracking
        self.seen_hashes = set()
        
        # Create output directories
        CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATS_DIR.mkdir(parents=True, exist_ok=True)
    
    def stream_and_batch(self, input_path: Path, batch_size: int) -> Iterator[List[str]]:
        """
        Stream file and yield batches of lines for parallel processing.
        
        Args:
            input_path: Path to input JSONL file
            batch_size: Number of lines per batch
            
        Yields:
            Batches of JSON lines
        """
        batch = []
        
        with open(input_path, 'r', encoding='utf-8', buffering=8*1024*1024) as f:  # 8MB buffer
            for line in f:
                batch.append(line)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            # Yield remaining lines
            if batch:
                yield batch
    
    def process_batches_parallel(self, input_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Process file in parallel batches.
        
        Args:
            input_path: Path to input file
            
        Yields:
            Processed records
        """
        logger.info(f"Processing with {self.num_workers} cores")
        
        # Configuration for workers
        config_dict = {
            'INCLUDE_CATEGORIES': self.config.INCLUDE_CATEGORIES,
            'STRIP_LATEX': self.config.STRIP_LATEX,
            'STRIP_HTML': self.config.STRIP_HTML,
            'REMOVE_URLS': self.config.REMOVE_URLS,
            'MIN_TITLE_LENGTH': self.config.MIN_TITLE_LENGTH,
            'MIN_ABSTRACT_LENGTH': self.config.MIN_ABSTRACT_LENGTH,
            'MAX_ABSTRACT_LENGTH': self.config.MAX_ABSTRACT_LENGTH,
            'MAX_TITLE_LENGTH': self.config.MAX_TITLE_LENGTH,
            'DEDUP_HASH_CHARS': self.config.DEDUP_HASH_CHARS,
            'TEXT_SEPARATOR': self.config.TEXT_SEPARATOR,
        }
        
        # Process in large batches for efficiency
        batch_size = self.config.BATCH_SIZE * self.num_workers
        
        for batch_num, lines_batch in enumerate(self.stream_and_batch(input_path, batch_size)):
            self.stats["total_records"] += len(lines_batch)
            
            # Process batch in parallel
            processed_records = process_batch_parallel(lines_batch, config_dict, self.num_workers)
            
            # Filter global duplicates
            final_records = []
            for record in processed_records:
                dedup_hash = record.pop('dedup_hash')  # Remove temp field
                
                if dedup_hash in self.seen_hashes:
                    self.stats["duplicate_records"] += 1
                    continue
                
                self.seen_hashes.add(dedup_hash)
                final_records.append(record)
                
                # Update stats
                self.stats["processed_records"] += 1
                self.stats["total_text_bytes"] += len(record['text'].encode('utf-8'))
                self.stats["avg_title_length"] += len(record['title'])
                self.stats["avg_abstract_length"] += len(record['abstract'])
                self.stats["avg_text_length"] += len(record['text'])
                
                # Count categories
                for cat in record['categories'].split():
                    self.stats["category_counts"][cat] += 1
            
            self.stats["skipped_records"] = (
                self.stats["total_records"] - 
                self.stats["processed_records"] - 
                self.stats["duplicate_records"]
            )
            
            # Log progress
            if batch_num % 10 == 0:
                logger.info(
                    f"Processed {self.stats['total_records']:,} records | "
                    f"Kept: {self.stats['processed_records']:,} | "
                    f"Skipped: {self.stats['skipped_records']:,} | "
                    f"Duplicates: {self.stats['duplicate_records']:,} | "
                    f"Speed: {len(lines_batch)} records/batch"
                )
            
            # Yield processed records
            yield from final_records
    
    def write_parquet_partitions(
        self,
        records_iterator: Iterator[Dict[str, Any]],
        output_dir: Path,
        partition_size_mb: int,
    ) -> int:
        """Write records to parquet partitions."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        partition_num = 0
        current_batch = []
        current_size_bytes = 0
        target_size_bytes = partition_size_mb * 1024 * 1024
        
        for record in records_iterator:
            # Estimate record size
            record_size = sum(
                len(str(v).encode('utf-8')) if v is not None else 0
                for v in record.values()
            )
            
            current_batch.append(record)
            current_size_bytes += record_size
            
            # Write partition when target size reached
            if current_size_bytes >= target_size_bytes:
                self._write_partition(current_batch, output_dir, partition_num)
                partition_num += 1
                current_batch = []
                current_size_bytes = 0
        
        # Write final partition
        if current_batch:
            self._write_partition(current_batch, output_dir, partition_num)
            partition_num += 1
        
        return partition_num
    
    def _write_partition(self, records: list, output_dir: Path, partition_num: int):
        """Write a partition to parquet file."""
        df = pd.DataFrame(records)
        
        output_file = output_dir / f"part-{partition_num:04d}.parquet"
        
        # Use pyarrow directly for better performance
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            output_file,
            compression=self.config.COMPRESSION,
            use_dictionary=True,
            write_statistics=True,
        )
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Written partition {partition_num:04d}: "
            f"{len(records):,} records, {file_size_mb:.2f} MB"
        )
    
    def create_sample(self, input_dir: Path, output_dir: Path, sample_size: int):
        """Create a sample dataset from processed partitions."""
        logger.info(f"Creating sample of {sample_size} records...")
        
        all_files = sorted(input_dir.glob("part-*.parquet"))
        
        if not all_files:
            logger.error("No partition files found!")
            return
        
        # Read and concatenate
        dfs = [pd.read_parquet(f) for f in all_files]
        df_full = pd.concat(dfs, ignore_index=True)
        
        # Sample randomly
        if len(df_full) > sample_size:
            df_sample = df_full.sample(
                n=sample_size,
                random_state=self.config.SAMPLE_RANDOM_SEED
            )
        else:
            df_sample = df_full
            logger.warning(
                f"Dataset only has {len(df_full)} records, "
                f"less than requested sample size {sample_size}"
            )
        
        # Write sample
        output_file = output_dir / "sample.parquet"
        table = pa.Table.from_pandas(df_sample)
        pq.write_table(
            table,
            output_file,
            compression=self.config.COMPRESSION,
        )
        
        logger.info(f"Sample created: {len(df_sample):,} records -> {output_file}")
    
    def save_statistics(self):
        """Save processing statistics to JSON."""
        # Finalize averages
        if self.stats["processed_records"] > 0:
            count = self.stats["processed_records"]
            self.stats["avg_title_length"] /= count
            self.stats["avg_abstract_length"] /= count
            self.stats["avg_text_length"] /= count
            self.stats["total_size_gb"] = self.stats["total_text_bytes"] / (1024**3)
        
        # Convert defaultdict to regular dict
        self.stats["category_counts"] = dict(self.stats["category_counts"])
        
        # Save to file
        stats_file = STATS_DIR / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved to: {stats_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records read:      {self.stats['total_records']:,}")
        logger.info(f"Successfully processed:  {self.stats['processed_records']:,}")
        logger.info(f"Skipped:                 {self.stats['skipped_records']:,}")
        logger.info(f"  - Duplicates:          {self.stats['duplicate_records']:,}")
        logger.info(f"Errors:                  {self.stats['error_records']:,}")
        logger.info(f"Total size:              {self.stats.get('total_size_gb', 0):.2f} GB")
        logger.info(f"Avg title length:        {self.stats['avg_title_length']:.1f} chars")
        logger.info(f"Avg abstract length:     {self.stats['avg_abstract_length']:.1f} chars")
        logger.info(f"Avg combined text:       {self.stats['avg_text_length']:.1f} chars")
        logger.info("="*60 + "\n")


def main():
    """Main processing pipeline with multiprocessing."""
    logger.info("Starting OPTIMIZED arXiv data processing pipeline")
    logger.info(f"Hardware: {mp.cpu_count()} CPU cores available")
    logger.info(f"Configuration: {ProcessingConfig.to_dict()}")
    
    # Initialize processor (will use all available cores)
    processor = OptimizedArxivDataProcessor()
    
    # Check input file exists
    if not RAW_DATA_PATH.exists():
        logger.error(f"Input file not found: {RAW_DATA_PATH}")
        return
    
    # Process and write partitions
    logger.info("Phase 1: Processing and partitioning data (multiprocessing enabled)...")
    start_time = datetime.now()
    
    records_stream = processor.process_batches_parallel(RAW_DATA_PATH)
    num_partitions = processor.write_parquet_partitions(
        records_stream,
        CLEAN_DATA_DIR,
        ProcessingConfig.PARTITION_SIZE_MB
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Created {num_partitions} partitions in {elapsed:.1f} seconds")
    logger.info(f"Processing speed: {processor.stats['total_records'] / elapsed:.0f} records/sec")
    
    # Create sample
    logger.info("Phase 2: Creating sample dataset...")
    processor.create_sample(
        CLEAN_DATA_DIR,
        SAMPLE_DATA_DIR,
        ProcessingConfig.SAMPLE_SIZE
    )
    
    # Save statistics
    logger.info("Phase 3: Saving statistics...")
    processor.save_statistics()
    
    logger.info("✓ Pipeline completed successfully!")


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('fork', force=True)
    main()
