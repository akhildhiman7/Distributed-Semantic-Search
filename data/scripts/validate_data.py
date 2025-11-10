"""
Data validation script - verify output quality and completeness.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from config import CLEAN_DATA_DIR, SAMPLE_DATA_DIR, STATS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate processed data quality."""
    
    def __init__(self):
        self.validation_results = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
        }
    
    def check_directories_exist(self) -> bool:
        """Check all output directories exist."""
        dirs_to_check = [CLEAN_DATA_DIR, SAMPLE_DATA_DIR, STATS_DIR]
        
        for dir_path in dirs_to_check:
            if not dir_path.exists():
                self.validation_results["checks_failed"].append(
                    f"Directory missing: {dir_path}"
                )
                return False
        
        self.validation_results["checks_passed"].append("All output directories exist")
        return True
    
    def check_partition_files(self) -> bool:
        """Check partition files exist and are valid."""
        partition_files = list(CLEAN_DATA_DIR.glob("part-*.parquet"))
        
        if not partition_files:
            self.validation_results["checks_failed"].append(
                "No partition files found in clean directory"
            )
            return False
        
        logger.info(f"Found {len(partition_files)} partition files")
        
        # Check each partition
        total_records = 0
        total_size_mb = 0
        
        for pf in partition_files:
            try:
                df = pd.read_parquet(pf)
                total_records += len(df)
                total_size_mb += pf.stat().st_size / (1024 * 1024)
                
                # Validate schema
                expected_columns = [
                    "paper_id", "title", "abstract", "categories",
                    "text", "text_length", "has_full_data"
                ]
                
                if not all(col in df.columns for col in expected_columns):
                    self.validation_results["checks_failed"].append(
                        f"Missing columns in {pf.name}"
                    )
                    return False
                
            except Exception as e:
                self.validation_results["checks_failed"].append(
                    f"Error reading {pf.name}: {e}"
                )
                return False
        
        self.validation_results["checks_passed"].append(
            f"All {len(partition_files)} partitions valid "
            f"({total_records:,} records, {total_size_mb:.2f} MB)"
        )
        
        # Check total size
        if total_size_mb < 100:
            self.validation_results["warnings"].append(
                f"Total size ({total_size_mb:.2f} MB) is less than 100 MB - "
                "might not meet 1GB requirement"
            )
        
        return True
    
    def check_sample_file(self) -> bool:
        """Check sample file exists and has correct size."""
        sample_file = SAMPLE_DATA_DIR / "sample.parquet"
        
        if not sample_file.exists():
            self.validation_results["checks_failed"].append(
                "Sample file not found"
            )
            return False
        
        try:
            df = pd.read_parquet(sample_file)
            
            if len(df) < 1000:
                self.validation_results["checks_failed"].append(
                    f"Sample has only {len(df)} records, expected ~10,000"
                )
                return False
            
            self.validation_results["checks_passed"].append(
                f"Sample file valid ({len(df):,} records)"
            )
            
            return True
            
        except Exception as e:
            self.validation_results["checks_failed"].append(
                f"Error reading sample file: {e}"
            )
            return False
    
    def check_data_quality(self) -> bool:
        """Check data quality on sample."""
        sample_file = SAMPLE_DATA_DIR / "sample.parquet"
        
        if not sample_file.exists():
            return False
        
        try:
            df = pd.read_parquet(sample_file)
            
            # Check for nulls
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.validation_results["warnings"].append(
                    f"Null values found: {null_counts[null_counts > 0].to_dict()}"
                )
            
            # Check text lengths
            if (df['text_length'] < 100).any():
                count = (df['text_length'] < 100).sum()
                self.validation_results["warnings"].append(
                    f"{count} records have text_length < 100"
                )
            
            # Check for duplicates
            if df['paper_id'].duplicated().any():
                count = df['paper_id'].duplicated().sum()
                self.validation_results["checks_failed"].append(
                    f"{count} duplicate paper_ids found"
                )
                return False
            
            # Check text field
            if df['text'].str.len().min() < 50:
                self.validation_results["warnings"].append(
                    "Some text fields are very short"
                )
            
            self.validation_results["checks_passed"].append(
                "Data quality checks passed on sample"
            )
            
            return True
            
        except Exception as e:
            self.validation_results["checks_failed"].append(
                f"Error checking data quality: {e}"
            )
            return False
    
    def check_statistics(self) -> bool:
        """Check statistics file exists and looks reasonable."""
        stat_files = list(STATS_DIR.glob("profile_*.json"))
        
        if not stat_files:
            self.validation_results["warnings"].append(
                "No statistics file found"
            )
            return True  # Not critical
        
        # Load most recent
        latest_stats = max(stat_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_stats) as f:
                stats = json.load(f)
            
            logger.info("\nStatistics Summary:")
            logger.info(f"  Total records: {stats.get('total_records', 0):,}")
            logger.info(f"  Processed: {stats.get('processed_records', 0):,}")
            logger.info(f"  Skipped: {stats.get('skipped_records', 0):,}")
            logger.info(f"  Duplicates: {stats.get('duplicate_records', 0):,}")
            logger.info(f"  Total size: {stats.get('total_size_gb', 0):.2f} GB")
            
            # Validate reasonable numbers
            if stats.get('processed_records', 0) < 10000:
                self.validation_results["warnings"].append(
                    "Fewer than 10,000 processed records - dataset may be too small"
                )
            
            if stats.get('total_size_gb', 0) < 0.5:
                self.validation_results["warnings"].append(
                    f"Total size {stats.get('total_size_gb', 0):.2f} GB is less than 1 GB target"
                )
            
            self.validation_results["checks_passed"].append(
                "Statistics file found and validated"
            )
            
            return True
            
        except Exception as e:
            self.validation_results["warnings"].append(
                f"Error reading statistics: {e}"
            )
            return True  # Not critical
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        logger.info("="*60)
        logger.info("DATA VALIDATION")
        logger.info("="*60)
        
        checks = [
            ("Directories", self.check_directories_exist),
            ("Partition files", self.check_partition_files),
            ("Sample file", self.check_sample_file),
            ("Data quality", self.check_data_quality),
            ("Statistics", self.check_statistics),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            logger.info(f"\nRunning: {check_name}...")
            try:
                result = check_func()
                if result:
                    logger.info(f"  ✓ {check_name} passed")
                else:
                    logger.error(f"  ✗ {check_name} failed")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ✗ {check_name} error: {e}")
                all_passed = False
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\n✓ Checks passed ({len(self.validation_results['checks_passed'])}):")
        for check in self.validation_results['checks_passed']:
            logger.info(f"  - {check}")
        
        if self.validation_results['warnings']:
            logger.info(f"\n⚠ Warnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                logger.info(f"  - {warning}")
        
        if self.validation_results['checks_failed']:
            logger.info(f"\n✗ Checks failed ({len(self.validation_results['checks_failed'])}):")
            for failure in self.validation_results['checks_failed']:
                logger.info(f"  - {failure}")
        
        logger.info("\n" + "="*60)
        
        if all_passed and not self.validation_results['checks_failed']:
            logger.info("✓ ALL VALIDATIONS PASSED")
        else:
            logger.error("✗ SOME VALIDATIONS FAILED")
        
        logger.info("="*60 + "\n")
        
        return all_passed


def main():
    """Run data validation."""
    validator = DataValidator()
    success = validator.run_all_checks()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
