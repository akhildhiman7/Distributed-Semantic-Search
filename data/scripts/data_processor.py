"""
Main data processing pipeline for arXiv dataset.
Handles ingestion, cleaning, deduplication, and partitioning.
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import sys

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


class ArxivDataProcessor:
    """High-performance, production-grade arXiv data processor."""
    
    def __init__(self, config: ProcessingConfig = ProcessingConfig):
        self.config = config
        self.text_cleaner = TextCleaner()
        
        # Statistics tracking
        self.stats = {
            "total_records": 0,
            "processed_records": 0,
            "skipped_records": 0,
            "duplicate_records": 0,
            "error_records": 0,
            "category_counts": defaultdict(int),
            "skip_reasons": defaultdict(int),
            "total_text_bytes": 0,
            "avg_title_length": 0,
            "avg_abstract_length": 0,
            "avg_text_length": 0,
        }
        
        # Deduplication tracking (hash of normalized title + first N chars of abstract)
        self.seen_hashes = set()
        
        # Create output directories
        CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _extract_paper_id(self, arxiv_id: str) -> int:
        """
        Convert arXiv ID to integer paper_id.
        
        Format: YYMM.NNNNN or arXiv:YYMM.NNNNN
        Example: "0704.0001" -> 7040001
        """
        try:
            # Remove "arXiv:" prefix if present
            arxiv_id = arxiv_id.replace("arXiv:", "")
            
            # Split on period
            parts = arxiv_id.split(".")
            if len(parts) != 2:
                return hash(arxiv_id) % (10**10)  # fallback to hash
            
            # Convert YYMM.NNNNN to integer
            yymm = parts[0].replace("/", "")  # handle old format like hep-ph/9901001
            nnnnn = parts[1]
            
            # For new format: 0704.0001 -> 7040001
            # For very old format with categories, use hash
            if len(yymm) <= 4 and yymm.isdigit():
                paper_id = int(yymm + nnnnn.zfill(5))
            else:
                paper_id = hash(arxiv_id) % (10**10)
            
            return paper_id
        except Exception as e:
            logger.warning(f"Error extracting paper_id from {arxiv_id}: {e}")
            return hash(arxiv_id) % (10**10)
    
    def _compute_dedup_hash(self, title: str, abstract: str) -> str:
        """Compute hash for deduplication based on normalized title + abstract prefix."""
        # Normalize: lowercase, strip whitespace
        title_norm = title.lower().strip()
        abstract_prefix = abstract.lower().strip()[:self.config.DEDUP_HASH_CHARS]
        
        combined = f"{title_norm}||{abstract_prefix}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _should_include_record(self, record: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Determine if record should be included based on filters.
        
        Returns:
            (should_include, skip_reason)
        """
        # Check required fields
        if not record.get("title"):
            return False, "missing_title"
        
        if not record.get("abstract"):
            return False, "missing_abstract"
        
        # Check categories filter
        if self.config.INCLUDE_CATEGORIES:
            categories = record.get("categories", "")
            # Check if any of the desired categories are present
            has_category = any(cat in categories for cat in self.config.INCLUDE_CATEGORIES)
            if not has_category:
                return False, "category_filtered"
        
        return True, None
    
    def _process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single arXiv record.
        
        Returns:
            Processed record dict or None if should be skipped
        """
        self.stats["total_records"] += 1
        
        try:
            # Check if record should be included
            should_include, skip_reason = self._should_include_record(record)
            if not should_include:
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"][skip_reason] += 1
                return None
            
            # Extract fields
            arxiv_id = record.get("id", "")
            title_raw = record.get("title", "")
            abstract_raw = record.get("abstract", "")
            categories = record.get("categories", "")
            
            # Clean text
            title_clean = self.text_cleaner.clean_text(
                title_raw,
                strip_latex=self.config.STRIP_LATEX,
                strip_html=self.config.STRIP_HTML,
                remove_urls=self.config.REMOVE_URLS,
            )
            
            abstract_clean = self.text_cleaner.clean_text(
                abstract_raw,
                strip_latex=self.config.STRIP_LATEX,
                strip_html=self.config.STRIP_HTML,
                remove_urls=self.config.REMOVE_URLS,
            )
            
            # Validate length
            if len(title_clean) < self.config.MIN_TITLE_LENGTH:
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"]["title_too_short"] += 1
                return None
            
            if len(abstract_clean) < self.config.MIN_ABSTRACT_LENGTH:
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"]["abstract_too_short"] += 1
                return None
            
            if len(abstract_clean) > self.config.MAX_ABSTRACT_LENGTH:
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"]["abstract_too_long"] += 1
                return None
            
            # Check for duplicates
            dedup_hash = self._compute_dedup_hash(title_clean, abstract_clean)
            if dedup_hash in self.seen_hashes:
                self.stats["duplicate_records"] += 1
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"]["duplicate"] += 1
                return None
            
            self.seen_hashes.add(dedup_hash)
            
            # Create combined text
            text = self.text_cleaner.create_search_text(
                title_clean,
                abstract_clean,
                separator=self.config.TEXT_SEPARATOR
            )
            
            # Validate final text
            is_valid, error_msg = self.text_cleaner.validate_text(
                text,
                min_length=self.config.MIN_ABSTRACT_LENGTH,
                max_length=self.config.MAX_ABSTRACT_LENGTH + self.config.MAX_TITLE_LENGTH
            )
            
            if not is_valid:
                self.stats["skipped_records"] += 1
                self.stats["skip_reasons"][f"validation_failed_{error_msg}"] += 1
                return None
            
            # Extract paper ID
            paper_id = self._extract_paper_id(arxiv_id)
            
            # Update statistics
            self.stats["processed_records"] += 1
            self.stats["total_text_bytes"] += len(text.encode('utf-8'))
            self.stats["avg_title_length"] += len(title_clean)
            self.stats["avg_abstract_length"] += len(abstract_clean)
            self.stats["avg_text_length"] += len(text)
            
            # Count categories
            for cat in categories.split():
                self.stats["category_counts"][cat] += 1
            
            # Return processed record
            return {
                "paper_id": paper_id,
                "title": title_clean,
                "abstract": abstract_clean,
                "categories": categories,
                "text": text,
                "text_length": len(text),
                "has_full_data": True,
            }
        
        except Exception as e:
            self.stats["error_records"] += 1
            logger.warning(f"Error processing record {record.get('id', 'unknown')}: {e}")
            return None
    
    def stream_records(self, input_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream records from JSONL file.
        
        Yields:
            Processed record dictionaries
        """
        logger.info(f"Starting to process: {input_path}")
        
        batch = []
        errors_in_batch = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON
                    record = json.loads(line.strip())
                    
                    # Process record
                    processed = self._process_record(record)
                    
                    if processed is not None:
                        batch.append(processed)
                    
                    # Yield batch when full
                    if len(batch) >= self.config.BATCH_SIZE:
                        yield from batch
                        batch = []
                        errors_in_batch = 0
                    
                    # Log progress
                    if line_num % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            f"Processed {line_num:,} lines | "
                            f"Kept: {self.stats['processed_records']:,} | "
                            f"Skipped: {self.stats['skipped_records']:,} | "
                            f"Duplicates: {self.stats['duplicate_records']:,}"
                        )
                
                except json.JSONDecodeError as e:
                    errors_in_batch += 1
                    if errors_in_batch > self.config.MAX_ERRORS_PER_BATCH:
                        logger.error(f"Too many errors in batch at line {line_num}, stopping")
                        break
                    continue
                
                except Exception as e:
                    logger.error(f"Unexpected error at line {line_num}: {e}")
                    errors_in_batch += 1
                    if errors_in_batch > self.config.MAX_ERRORS_PER_BATCH:
                        break
                    continue
        
        # Yield remaining records
        if batch:
            yield from batch
    
    def write_parquet_partitions(
        self,
        records_iterator: Iterator[Dict[str, Any]],
        output_dir: Path,
        partition_size_mb: int,
    ) -> int:
        """
        Write records to parquet partitions.
        
        Returns:
            Number of partitions written
        """
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
        
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression=self.config.COMPRESSION,
            index=False,
        )
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Written partition {partition_num:04d}: "
            f"{len(records):,} records, {file_size_mb:.2f} MB"
        )
    
    def create_sample(self, input_dir: Path, output_dir: Path, sample_size: int):
        """Create a sample dataset from processed partitions."""
        logger.info(f"Creating sample of {sample_size} records...")
        
        # Read all partitions
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
        df_sample.to_parquet(
            output_file,
            engine='pyarrow',
            compression=self.config.COMPRESSION,
            index=False,
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
        
        # Convert defaultdict to regular dict for JSON serialization
        self.stats["category_counts"] = dict(self.stats["category_counts"])
        self.stats["skip_reasons"] = dict(self.stats["skip_reasons"])
        
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
    """Main processing pipeline."""
    logger.info("Starting arXiv data processing pipeline")
    logger.info(f"Configuration: {ProcessingConfig.to_dict()}")
    
    # Initialize processor
    processor = ArxivDataProcessor()
    
    # Check input file exists
    if not RAW_DATA_PATH.exists():
        logger.error(f"Input file not found: {RAW_DATA_PATH}")
        return
    
    # Process and write partitions
    logger.info("Phase 1: Processing and partitioning data...")
    records_stream = processor.stream_records(RAW_DATA_PATH)
    num_partitions = processor.write_parquet_partitions(
        records_stream,
        CLEAN_DATA_DIR,
        ProcessingConfig.PARTITION_SIZE_MB
    )
    
    logger.info(f"Created {num_partitions} partitions in {CLEAN_DATA_DIR}")
    
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
    main()
