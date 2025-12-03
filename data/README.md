# Data Engineering - Member 1

## Overview

High-quality data processing pipeline for arXiv metadata. Produces clean, deduplicated, ready-to-embed dataset with robust provenance tracking.

## Structure

```
data/
├── scripts/
│   ├── config.py              # Configuration parameters
│   ├── text_cleaner.py        # Text cleaning utilities
│   ├── data_processor.py      # Main processing pipeline
│   ├── validate_data.py       # Data validation script
│   └── requirements.txt       # Python dependencies
├── clean/                     # Cleaned parquet partitions (output)
├── sample/                    # 10k sample dataset (output)
└── stats/                     # Processing logs & statistics (output)
```

## Quick Start

### 1. Install Dependencies

```bash
cd data/scripts
pip install -r requirements.txt
```

### 2. Run Processing Pipeline

**RECOMMENDED: Use the optimized multiprocessing version for M2/M3 Macs:**

```bash
# Optimized for Apple Silicon (8-10 cores) - 5-8x faster!
python data_processor_optimized.py
```

**Or use the standard single-core version:**

```bash
python data_processor.py
```

The optimized version will:

- Stream-parse the arXiv JSONL file
- Clean and normalize text (remove LaTeX, HTML, normalize whitespace)
- Deduplicate by title + abstract hash
- Filter by categories (ML/AI/CS focus)
- Write parquet partitions (~150 MB each)
- Create 10k sample dataset
- Generate statistics and logs

### 3. Validate Output

```bash
python validate_data.py
```

## Configuration

All parameters are in `config.py`. Key settings:

```python
# Data quality
MIN_ABSTRACT_LENGTH = 100      # characters
MIN_TITLE_LENGTH = 10          # characters

# Categories (ML/AI/CS focus for better semantic quality)
INCLUDE_CATEGORIES = [
    "cs.LG", "cs.AI", "cs.CL", "cs.CV",
    "cs.IR", "cs.NE", "stat.ML"
]

# Partitioning
PARTITION_SIZE_MB = 150        # target size per file
COMPRESSION = "snappy"         # parquet compression

# Sampling
SAMPLE_SIZE = 10000            # records for quick integration
SAMPLE_RANDOM_SEED = 42        # reproducibility
```

## Output Schema

Parquet files contain:

| Field           | Type   | Description                             |
| --------------- | ------ | --------------------------------------- |
| `paper_id`      | int64  | Unique ID (derived from arXiv ID)       |
| `title`         | string | Cleaned paper title                     |
| `abstract`      | string | Cleaned abstract                        |
| `categories`    | string | arXiv categories (space-separated)      |
| `text`          | string | Combined title + abstract for embedding |
| `text_length`   | int32  | Length of combined text                 |
| `has_full_data` | bool   | Quality flag                            |

## Data Quality Features

### Text Cleaning

- ✅ LaTeX command removal (`\command{text}` → `text`)
- ✅ HTML tag stripping and entity decoding
- ✅ URL and email removal
- ✅ Unicode normalization to ASCII
- ✅ Whitespace normalization (collapse multiple spaces)

### Deduplication

- MD5 hash of normalized title + first 200 chars of abstract
- Catches near-duplicates from paper revisions

### Validation

- Length checks (min/max for title and abstract)
- Content validation (sufficient alphabetic characters)
- Category filtering (ML/AI/CS focus)

## Statistics Output

After processing, check `data/stats/` for:

- `profile_<timestamp>.json` - Detailed statistics
- `processing_<timestamp>.log` - Full processing log

Statistics include:

- Total/processed/skipped/duplicate counts
- Category distribution
- Skip reason breakdown
- Average text lengths
- Total dataset size

## Performance

### Optimized Version (data_processor_optimized.py)

On MacBook Air M2 (8 cores):

- **Processing speed**: ~40,000-60,000 records/second
- **Memory usage**: ~3-4 GB (multiprocessing + streaming)
- **Output size**: ~1-1.5 GB (after filtering and compression)
- **Duration**: 3-5 minutes for full 4.6 GB dataset
- **Speedup**: 6-8x faster than single-core

### Standard Version (data_processor.py)

- **Processing speed**: ~5,000-10,000 records/second
- **Memory usage**: <2 GB (streaming architecture)
- **Duration**: 10-20 minutes for full dataset

### Benchmark Your System

```bash
python benchmark.py  # Test on 10k records, see estimated time
```

## Reproducibility

All processing is deterministic and reproducible:

1. Fixed random seed for sampling
2. Consistent deduplication (hash-based)
3. Version-controlled configuration
4. Logged processing parameters

## Integration Points

### For Member 2 (Embeddings)

- **Sample dataset**: `data/sample/sample.parquet` (10k records, ready Day 2)
- **Full dataset**: `data/clean/part-*.parquet` (all partitions, ready Day 3-4)
- **Schema**: See table above - `text` field is ready for embedding

### For Member 3 (Milvus)

- Parquet files can be directly loaded into pandas/pyarrow
- `paper_id` field is the primary key (INT64)
- Category field enables metadata filtering

## Validation Checklist

✅ **Data Quality**

- [ ] No empty titles or abstracts
- [ ] All text lengths within bounds
- [ ] No duplicate papers
- [ ] Valid UTF-8 encoding

✅ **Output Files**

- [ ] `data/clean/part-*.parquet` exists (multiple files)
- [ ] `data/sample/sample.parquet` exists (10k records)
- [ ] `data/stats/profile_*.json` exists

✅ **Statistics**

- [ ] Total size ≥ 1 GB
- [ ] Processed count > 100k records
- [ ] Category distribution shows ML/AI/CS focus
- [ ] Low error rate (<0.1%)

## Troubleshooting

### Memory issues

- Reduce `BATCH_SIZE` in config
- Reduce `PARTITION_SIZE_MB` for more, smaller files

### Processing too slow

- Increase `BATCH_SIZE`
- Disable verbose logging (reduce `LOG_INTERVAL`)

### Too few records

- Expand `INCLUDE_CATEGORIES` (or set to empty list for all)
- Lower `MIN_ABSTRACT_LENGTH`

### Parsing errors

- Check log file in `data/stats/`
- Invalid JSON lines are automatically skipped
- Increase `MAX_ERRORS_PER_BATCH` if needed

## Next Steps

After completing data processing:

1. ✅ Verify sample.parquet loads correctly
2. ✅ Share statistics with team
3. ✅ Provide sample to Member 2 for embedding testing
4. ✅ Document any data quality issues found

## Contact

Member 1 - Data Engineer
