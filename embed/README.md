# Embedding Generation - Member 2

## Overview

High-performance embedding generation pipeline using Sentence-BERT (all-MiniLM-L6-v2). Optimized for M2 MacBook Air with Metal Performance Shaders (MPS) acceleration.

## Structure

```
embed/
├── config.py                    # Configuration (batch size, device, etc.)
├── embedding_generator.py       # Main pipeline
├── test_embeddings.py          # Quality validation
├── requirements.txt            # Dependencies
├── embeddings/                 # Output embeddings (created)
│   ├── sample/                 # Sample dataset embeddings
│   │   ├── sample_embeddings.npy
│   │   └── sample_metadata.parquet
│   └── full/                   # Full dataset embeddings
│       ├── part-0000_embeddings.npy
│       ├── part-0000_metadata.parquet
│       └── ...
├── reports/                    # Processing logs & stats
└── model_cache/                # Downloaded model cache
```

## Quick Start

### 1. Install Dependencies

```bash
cd embed
pip install -r requirements.txt
```

This installs:

- `sentence-transformers` - Pre-trained semantic models
- `torch` - Deep learning framework (with MPS support)
- Model dependencies

### 2. Run Embedding Generation

#### Test on Sample (10k records, ~30 seconds)

```bash
python embedding_generator.py
```

This will:

1. Load all-MiniLM-L6-v2 model (133 MB download on first run)
2. Process sample dataset using MPS (Apple Silicon GPU)
3. Generate 384-dimensional embeddings
4. Save embeddings + metadata
5. Validate output quality

#### Process Full Dataset

The script automatically processes both sample and full datasets if available.

### 3. Validate Output

```bash
python test_embeddings.py
```

## Configuration

Key settings in `config.py`:

```python
# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# M2 Optimization
BATCH_SIZE = 128          # Optimal for M2 (can go to 256)
USE_MPS = True            # Use Apple Silicon GPU
NUM_WORKERS = 6           # Parallel workers

# Memory
USE_MEMMAP = True         # Memory-mapped arrays (large datasets)
MAX_SEQ_LENGTH = 512      # Max tokens per text

# Quality
NORMALIZE_EMBEDDINGS = True   # For cosine similarity
VALIDATE_EMBEDDINGS = True    # Check output quality
```

## Output Format

### Embeddings (.npy files)

```python
import numpy as np

# Load embeddings
embeddings = np.load("embeddings/sample/sample_embeddings.npy")
# Shape: (10000, 384) for sample
# dtype: float32
```

### Metadata (.parquet files)

```python
import pandas as pd

# Load metadata
df = pd.read_parquet("embeddings/sample/sample_metadata.parquet")

# Columns:
# - paper_id (int64)
# - title (string)
# - abstract (string)
# - categories (string)
# - text (string)
# - text_length (int32)
# - embedding_index (int64) - row number for alignment
# - embedding_norm (float64) - L2 norm of embedding
```

## Performance

### Expected Performance on M2 MacBook Air

| Dataset | Records | Time    | Throughput     |
| ------- | ------- | ------- | -------------- |
| Sample  | 10,000  | ~30s    | ~333 texts/sec |
| Full    | 510,000 | ~25 min | ~340 texts/sec |

**Optimization factors:**

- ✅ MPS (Metal) acceleration: 3-5x faster than CPU
- ✅ Batched inference: 128 texts at once
- ✅ Memory-mapped I/O: No memory overflow
- ✅ Unified memory: Efficient CPU-GPU transfer

### Resource Usage

- **Memory**: ~1-2 GB (peak)
- **Disk**: ~2 GB (embeddings) + 0.6 GB (metadata)
- **GPU**: M2 cores at 80-100%

## Model Details

### all-MiniLM-L6-v2

- **Size**: 80M parameters (133 MB)
- **Architecture**: 6-layer BERT
- **Output**: 384-dimensional dense vectors
- **Performance**: High quality, fast inference
- **Use case**: Semantic search, similarity, clustering

### Embedding Properties

- **Normalized**: Unit vectors (cosine similarity = dot product)
- **Dimension**: 384 (optimal balance of quality/speed)
- **Range**: [-1, 1] (after normalization)

## Quality Checks

Automatic validation:

- ✅ Shape correctness (N x 384)
- ✅ No NaN or Inf values
- ✅ Norm bounds (0.1 < norm < 10.0)
- ✅ Row alignment with metadata

## Integration Points

### For Member 3 (Milvus)

```python
# Load embeddings
embeddings = np.load("embeddings/full/part-0000_embeddings.npy")
metadata = pd.read_parquet("embeddings/full/part-0000_metadata.parquet")

# Verify alignment
assert len(embeddings) == len(metadata)
assert np.allclose(
    np.linalg.norm(embeddings, axis=1),
    metadata['embedding_norm'].values
)

# Insert to Milvus
for i, row in metadata.iterrows():
    milvus_collection.insert([
        [row['paper_id']],
        [embeddings[i].tolist()],
        [row['title']],
        [row['abstract']],
        [row['categories']],
    ])
```

### For Member 4 (API)

The same model is used for query encoding:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode user query
query = "neural networks for image classification"
query_embedding = model.encode(query, normalize_embeddings=True)

# Search in Milvus
results = milvus_collection.search(
    data=[query_embedding.tolist()],
    anns_field="vector",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=5
)
```

## Troubleshooting

### MPS Not Available

If MPS fails, it automatically falls back to CPU. Check:

```python
import torch
print(torch.backends.mps.is_available())  # Should be True on M2
```

### Out of Memory

Reduce batch size in `config.py`:

```python
BATCH_SIZE = 64  # or 32
```

### Slow Processing

- Ensure MPS is enabled (check logs for "Using Metal Performance Shaders")
- Close other apps to free GPU
- Increase batch size to 256 if you have 16GB RAM

### Model Download Fails

Set cache directory:

```python
CACHE_DIR = Path.home() / ".cache" / "sentence_transformers"
```

## Validation Checklist

✅ **Model Setup**

- [ ] Model downloaded successfully
- [ ] MPS/GPU detected and enabled
- [ ] Test embedding generated

✅ **Output Files**

- [ ] `embeddings/sample/sample_embeddings.npy` exists
- [ ] `embeddings/sample/sample_metadata.parquet` exists
- [ ] `embeddings/full/part-*_embeddings.npy` exists (9 files)
- [ ] `reports/speed_report_*.json` exists

✅ **Quality**

- [ ] All embeddings have shape (N, 384)
- [ ] No NaN or Inf values
- [ ] Embedding norms in valid range
- [ ] Row count matches metadata count

✅ **Performance**

- [ ] Sample: >300 texts/sec
- [ ] Full: >300 texts/sec
- [ ] Total time < 30 minutes

## Next Steps

1. ✅ Verify sample embeddings load correctly
2. ✅ Check alignment with metadata (row order)
3. ✅ Share statistics with team
4. ✅ Provide sample to Member 3 for Milvus testing

## Statistics

Check `reports/speed_report_*.json` for:

- Total texts processed
- Processing time
- Throughput (texts/sec)
- Device used (MPS/CPU)
- Per-partition stats

## Contact

Member 2 - ML Engineer
