"""
Quick benchmark to compare single-core vs multi-core performance.
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_processor import ArxivDataProcessor
from data_processor_optimized import OptimizedArxivDataProcessor
from config import RAW_DATA_PATH


def benchmark_processing(num_lines: int = 10000):
    """Benchmark both processors on a subset of data."""
    print("="*60)
    print(f"BENCHMARKING PROCESSORS ({num_lines:,} records)")
    print("="*60 + "\n")
    
    if not RAW_DATA_PATH.exists():
        print(f"Error: Input file not found: {RAW_DATA_PATH}")
        return
    
    # Read test lines
    print(f"Loading {num_lines:,} records from file...")
    test_lines = []
    with open(RAW_DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            test_lines.append(line)
    
    print(f"Loaded {len(test_lines):,} records\n")
    
    # Test optimized processor
    print("Testing OPTIMIZED processor (multiprocessing)...")
    start = time.time()
    
    processor_opt = OptimizedArxivDataProcessor()
    config_dict = {
        'INCLUDE_CATEGORIES': processor_opt.config.INCLUDE_CATEGORIES,
        'STRIP_LATEX': processor_opt.config.STRIP_LATEX,
        'STRIP_HTML': processor_opt.config.STRIP_HTML,
        'REMOVE_URLS': processor_opt.config.REMOVE_URLS,
        'MIN_TITLE_LENGTH': processor_opt.config.MIN_TITLE_LENGTH,
        'MIN_ABSTRACT_LENGTH': processor_opt.config.MIN_ABSTRACT_LENGTH,
        'MAX_ABSTRACT_LENGTH': processor_opt.config.MAX_ABSTRACT_LENGTH,
        'MAX_TITLE_LENGTH': processor_opt.config.MAX_TITLE_LENGTH,
        'DEDUP_HASH_CHARS': processor_opt.config.DEDUP_HASH_CHARS,
        'TEXT_SEPARATOR': processor_opt.config.TEXT_SEPARATOR,
    }
    
    from data_processor_optimized import process_batch_parallel
    results_opt = process_batch_parallel(test_lines, config_dict, processor_opt.num_workers)
    
    time_opt = time.time() - start
    speed_opt = len(test_lines) / time_opt
    
    print(f"  Time: {time_opt:.2f} seconds")
    print(f"  Speed: {speed_opt:.0f} records/sec")
    print(f"  Processed: {len(results_opt):,} records")
    print(f"  Using: {processor_opt.num_workers} cores\n")
    
    # Print results
    print("="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Optimized (multicore):  {speed_opt:.0f} records/sec")
    print(f"Cores used: {processor_opt.num_workers}")
    print(f"Speedup: ~{processor_opt.num_workers}x theoretical")
    print("="*60 + "\n")
    
    print("✓ For the full 4.6GB dataset:")
    print(f"  Estimated time: {(2.4e6 / speed_opt / 60):.1f} minutes")
    print(f"  (assuming ~2.4M records in dataset)")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)
    
    # Benchmark on 10k records
    benchmark_processing(10000)
