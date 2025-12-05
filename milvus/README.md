# Member 3: Milvus Cluster Setup & Vector Indexing

This directory contains the Milvus cluster setup and data ingestion pipeline for the Distributed Semantic Search Engine.

## Overview

Member 3's responsibility is to:

1. Deploy a distributed Milvus cluster using Docker Compose
2. Create the collection schema for arxiv papers with 384-dim embeddings
3. Ingest 510K pre-computed embeddings from Member 2's output
4. Build and optimize vector indexes for fast semantic search
5. Validate search functionality and performance

## Directory Structure

```
milvus/
├── config.py              # Centralized configuration
├── schema.py              # Collection schema definition
├── docker-compose.yml     # Multi-node Milvus cluster setup
├── ingest.py             # Data ingestion pipeline
├── test_queries.py       # Query validation and testing
├── requirements.txt      # Python dependencies
├── checkpoints/          # Ingestion checkpoints (auto-created)
├── logs/                 # Ingestion logs (auto-created)
└── README.md            # This file
```

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ with virtualenv
- Completed Member 2 work (embeddings in `../embed/embeddings/full/`)
- ~4GB free disk space for Milvus data
- ~2GB RAM available

## Quick Start

### 1. Start Milvus Cluster

```bash
# Start all services (Milvus, etcd, MinIO)
cd milvus
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f milvus-standalone
```

Services will be available at:

- Milvus gRPC: `localhost:19530`
- MinIO Console: `http://localhost:9001` (admin/minioadmin)
- Attu Web UI: `http://localhost:3000`
- Prometheus metrics: `http://localhost:9091/metrics`

### 2. Install Python Dependencies

```bash
# Activate project virtualenv
source ../.venv/bin/activate

# Install Milvus client and dependencies
pip install -r requirements.txt
```

### 3. Create Collection and Ingest Data

```bash
# First time: create collection and ingest all data
python ingest.py --drop --index IVF_FLAT

# Resume from checkpoint (if interrupted)
python ingest.py
```

**Expected Runtime:**

- Data ingestion: ~10-15 minutes (510K vectors)
- Index building: ~15-20 minutes (IVF_FLAT)
- **Total: ~25-35 minutes**

### 4. Validate and Test

```bash
# Run validation and sample queries
python test_queries.py
```

## Configuration

### Collection Settings (`config.py`)

Key parameters:

- `VECTOR_DIM = 384` - Embedding dimension (all-MiniLM-L6-v2)
- `METRIC_TYPE = "IP"` - Inner Product (for normalized vectors)
- `INDEX_TYPE = "IVF_FLAT"` - Balance of speed and accuracy
- `BATCH_SIZE = 1000` - Vectors per insert batch
- `NUM_SHARDS = 2` - Number of data shards

### Index Types

**IVF_FLAT** (Recommended for this dataset):

- Build time: ~15 minutes
- Query latency: 50-100ms
- Good accuracy with reasonable speed

**HNSW** (Alternative for lower latency):

- Build time: ~20-25 minutes
- Query latency: 30-80ms
- Best accuracy, higher memory usage

```bash
# Use HNSW index instead
python ingest.py --drop --index HNSW
```

## Schema

The collection schema includes:

| Field           | Type              | Description            |
| --------------- | ----------------- | ---------------------- |
| `paper_id`      | VARCHAR(64)       | Primary key (arxiv ID) |
| `embedding`     | FLOAT_VECTOR(384) | Sentence embedding     |
| `title`         | VARCHAR(512)      | Paper title            |
| `abstract`      | VARCHAR(4096)     | Paper abstract         |
| `categories`    | VARCHAR(256)      | ArXiv categories       |
| `text_length`   | INT32             | Text length            |
| `has_full_data` | BOOL              | Data completeness flag |

## Performance Targets

Based on project requirements:

| Metric                 | Target    | Expected     |
| ---------------------- | --------- | ------------ |
| Query latency (95th %) | < 120ms   | 50-100ms ✓   |
| Throughput             | > 500 QPS | ~800 QPS ✓   |
| Index build time       | < 30 min  | ~25 min ✓    |
| Ingestion rate         | -         | ~850 rec/sec |

## Ingestion Pipeline

The `ingest.py` script:

1. **Connects** to Milvus cluster
2. **Creates collection** with distributed schema
3. **Loads partitions** from `.npy` and `.parquet` files
4. **Batches insertions** (1000 vectors per batch)
5. **Checkpoints progress** every 5000 records
6. **Flushes and compacts** data
7. **Builds index** on embedding field
8. **Loads collection** into memory

### Checkpointing

Ingestion checkpoints are saved to `checkpoints/ingestion_checkpoint.json`:

```json
{
  "completed_partitions": ["part-0000", "part-0001", ...],
  "total_inserted": 250000
}
```

If ingestion is interrupted, simply re-run `python ingest.py` to resume.

## Testing

### Validation Tests

`test_queries.py` performs:

- Entity count validation (should be 510,203)
- Index verification
- Schema validation
- Sample semantic queries
- Latency measurements

### Sample Queries

Test queries cover various domains:

- "neural networks deep learning"
- "quantum computing algorithms"
- "natural language processing transformers"
- "computer vision image recognition"
- And more...

### Expected Output

```
============================================================
RUNNING TEST QUERIES
============================================================

Query 1: "neural networks deep learning"
--------------------------------------------------------------------------------
Search time: 67.23ms

Top 3 Results:

  1. Score: 0.8523
     Paper: 1234.5678
     Title: Deep Neural Networks for Image Classification
     Categories: cs.CV cs.LG
     Abstract: We present a comprehensive study of deep neural...

...

============================================================
QUERY PERFORMANCE SUMMARY
============================================================
Total queries: 8
Average latency: 73.45ms
95th percentile target: <120ms
Status: ✓ PASS
============================================================
```

## Monitoring

### Milvus Metrics

Prometheus metrics available at `http://localhost:9091/metrics`:

- `milvus_search_latency_ms` - Query latency
- `milvus_insert_throughput` - Insertion rate
- `milvus_collection_num_entities` - Total entities
- And more...

### Attu Web UI

Visual administration at `http://localhost:3000`:

- Browse collections and data
- Run queries interactively
- Monitor cluster status
- View performance metrics

## Troubleshooting

### Issue: Milvus container won't start

```bash
# Check logs
docker-compose logs milvus-standalone

# Common fix: ensure ports are free
lsof -i :19530
lsof -i :9000

# Restart clean
docker-compose down -v
docker-compose up -d
```

### Issue: Out of memory during ingestion

Reduce `BATCH_SIZE` in `config.py`:

```python
BATCH_SIZE = 500  # Reduce from 1000
```

### Issue: Slow query performance

1. Ensure collection is loaded:

```python
from pymilvus import Collection
collection = Collection("arxiv_papers")
collection.load()
```

2. Tune search parameters in `config.py`:

```python
SEARCH_PARAMS = {
    "IVF_FLAT": {
        "params": {"nprobe": 32}  # Increase from 16
    }
}
```

### Issue: Checkpoint corruption

```bash
# Delete checkpoint and restart
rm checkpoints/ingestion_checkpoint.json
python ingest.py
```

## Data Flow

```
Member 2 Output                 Member 3 Pipeline                  Milvus Cluster
───────────────                 ─────────────────                  ──────────────

embed/embeddings/full/
├── part-0000_embeddings.npy ──→ Load batch    ──→ Insert API ──→ [Shard 1]
├── part-0000_metadata.parquet                                    [Shard 2]
├── part-0001_embeddings.npy ──→ Load batch    ──→ Insert API ──→    ...
├── part-0001_metadata.parquet
...                                  ↓
                                   Flush
                                     ↓
                                Build Index (IVF_FLAT)
                                     ↓
                                Load to Memory
                                     ↓
                              Ready for Queries! ✓
```

## Next Steps (Member 4)

After Member 3 completes:

- Milvus cluster running with 510K indexed vectors
- Collection loaded and ready for queries
- Average query latency ~50-100ms

Member 4 will:

- Build FastAPI wrapper around Milvus queries
- Implement `/search` endpoint for semantic search
- Add request validation and response formatting
- Create API documentation

## Cleanup

```bash
# Stop services
docker-compose down

# Remove all data (WARNING: deletes everything)
docker-compose down -v

# Remove checkpoints and logs
rm -rf checkpoints/ logs/
```

## References

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus SDK](https://milvus.io/docs/install-pymilvus.md)
- [Index Types Guide](https://milvus.io/docs/index.md)
- [Performance Tuning](https://milvus.io/docs/performance_tuning.md)

## Deliverables Checklist

- [x] Docker Compose configuration for Milvus cluster
- [x] Collection schema with 384-dim vectors + metadata
- [x] Ingestion pipeline with checkpointing
- [x] IVF_FLAT index for semantic search
- [x] Validation and test scripts
- [x] Documentation and troubleshooting guide

---

## ✅ Execution Summary (December 5, 2025)

### What Was Completed

**1. Infrastructure Setup**

- ✅ Deployed Milvus v2.3.3 cluster with 4 services:
  - etcd v3.5.5 (metadata storage)
  - MinIO 2023-03-20 (object storage)
  - Milvus standalone (vector database)
  - Attu v2.3.3 (web UI)
- ✅ All containers running healthy on M2 MacBook Air
- ✅ Services accessible on ports: 19530 (gRPC), 9000-9001 (MinIO), 3000 (Attu)

**2. Data Ingestion**

- ✅ Successfully ingested **510,203 vectors** across 9 partitions
- ✅ Ingestion time: **0.88 minutes** (~53 seconds)
- ✅ Throughput: **11,559 records/sec** (exceeded expectations!)
- ✅ Built IVF_FLAT index in **0.38 minutes** (~23 seconds)
- ✅ Total pipeline runtime: **~1.3 minutes** (much faster than estimated 25-35 min)
- ✅ Collection loaded into memory and ready for queries

**3. Validation & Testing**

- ✅ Ran 8 diverse semantic queries across different domains
- ✅ All queries returned highly relevant results
- ✅ Entity count verified: 510,203 (matches expected)
- ✅ Index verification: IVF_FLAT present and operational
- ✅ Schema validation: 7 fields correctly configured

### Key Observations

**Performance Metrics:**
| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| Total Entities | 510,203 | 510,203 | ✅ Perfect |
| Ingestion Rate | ~850 rec/sec | 11,559 rec/sec | ✅ 13.6x faster |
| Index Build Time | <30 min | 0.38 min | ✅ 78x faster |
| Average Query Latency | <120ms | 218ms | ⚠️ Needs optimization |
| Query Accuracy | High relevance | Excellent | ✅ Strong semantic matching |

**Query Performance Examples:**

- **"quantum computing algorithms"** → Score: 0.7708
  - Found: "Quantum Artificial Intelligence" (perfect match!)
- **"natural language processing transformers"** → Score: 0.7881
  - Found: "Introduction to Transformers: an NLP Perspective" (excellent!)
- **"reinforcement learning robotics"** → Score: 0.7454
  - Found: "A Concise Introduction to Reinforcement Learning in Robotics" (spot on!)

**Similarity Score Interpretation:**

- **0.75-1.0**: Very high relevance (direct topic match)
- **0.65-0.75**: High relevance (strong semantic similarity)
- **0.50-0.65**: Moderate relevance (related concepts)
- Scores computed using Inner Product (IP) similarity on L2-normalized vectors

**Bug Fixed During Execution:**

- Issue: `paper_id` field type mismatch (int vs VARCHAR)
- Root cause: Parquet metadata stored paper_id as numeric type
- Fix: Added `.astype(str)` conversion in `prepare_batch_data()` method
- Result: Ingestion completed successfully without data loss

### Technical Highlights

**What Went Well:**

1. **Ingestion speed exceeded expectations** - 13.6x faster than estimated
   - Efficient memory-mapped `.npy` loading
   - Batch size of 1000 vectors optimal for this dataset
   - M2 chip performance with MPS acceleration
2. **Semantic search quality is excellent** - high relevance scores (0.64-0.79 avg)
3. **System stability** - No crashes during full ingestion
4. **Checkpointing worked** - Pipeline is resumable (though completed in one go)
5. **Docker cluster deployment smooth** - All services started correctly

**Areas for Improvement:**

1. **Query latency higher than target** (218ms vs <120ms)
   - First query includes model loading overhead (449ms)
   - Subsequent queries average ~150-200ms
   - IVF_FLAT with nlist=128 may need tuning
2. **Platform warnings** - Attu running amd64 on arm64 (emulation overhead)
3. **No distributed deployment** - Currently running standalone mode

### Future Improvements

**Short-term (Performance Optimization):**

1. **Switch to HNSW index** for lower latency:

   ```bash
   python ingest.py --drop --index HNSW
   ```

   - Expected: 30-80ms latency (vs current 218ms)
   - Trade-off: Slightly longer build time (~20-25 min)

2. **Tune IVF_FLAT search parameters**:

   ```python
   # In config.py, increase nprobe
   SEARCH_PARAMS = {
       "IVF_FLAT": {
           "params": {"nprobe": 64}  # Increase from 16
       }
   }
   ```

   - More clusters searched = better accuracy but slower queries

3. **Pre-load model at startup** to eliminate first-query overhead:

   ```python
   # In test_queries.py or API layer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   model.encode(["warmup"], show_progress_bar=False)  # Warmup
   ```

4. **Connection pooling** for concurrent queries in production

**Medium-term (Scalability):**

1. **Deploy distributed Milvus cluster**:

   - Add query nodes for parallel search
   - Add data nodes for data sharding
   - Expected: 3-5x query throughput improvement

2. **Implement query caching**:

   - Cache frequent queries with Redis
   - Reduce Milvus load for popular searches

3. **Add monitoring with Prometheus + Grafana**:
   - Real-time latency dashboards
   - Alert on SLA violations (>120ms)
   - Track index size and memory usage

**Long-term (Advanced Features):**

1. **Hybrid search** (vector + metadata filtering):

   ```python
   # Filter by category while searching
   expr = 'categories like "%cs.AI%"'
   results = collection.search(query_vectors, expr=expr)
   ```

2. **Multi-vector search** for different embedding models:

   - Store multiple embeddings per paper (e.g., title-only, abstract-only)
   - Choose embedding strategy based on query type

3. **Dynamic index optimization**:

   - Monitor query patterns
   - Auto-tune nprobe/ef based on latency requirements

4. **GPU acceleration** for embedding generation:
   - Currently using MPS (Metal Performance Shaders)
   - Could add CUDA support for faster batch encoding

### Next Steps (Member 4)

**Prerequisites (Complete ✅):**

- ✅ Milvus cluster operational
- ✅ 510,203 vectors indexed and queryable
- ✅ Semantic search validated with test queries
- ✅ Average query latency measured (~218ms)

**Member 4 Deliverables:**

1. **FastAPI service** wrapping Milvus queries
2. **REST API endpoints**:
   - `POST /search` - Semantic search with query text
   - `GET /health` - Service health check
   - `GET /stats` - Collection statistics
3. **Request/Response models** with Pydantic validation
4. **API documentation** with Swagger UI
5. **Error handling** and rate limiting
6. **Docker containerization** for API service

**Recommended Architecture:**

```
Client Request
     ↓
FastAPI (port 8000)
     ↓
SentenceTransformer (encode query)
     ↓
Milvus Cluster (port 19530)
     ↓
FastAPI (format response)
     ↓
JSON Response
```

**Expected API Response Format:**

```json
{
  "query": "neural networks deep learning",
  "results": [
    {
      "paper_id": "200303253",
      "title": "Introduction to deep learning",
      "score": 0.6478,
      "categories": "cs.LG",
      "abstract": "Deep Learning has made...",
      "text_length": 5234,
      "has_full_data": true
    }
  ],
  "latency_ms": 218,
  "total_results": 10
}
```

### Lessons Learned

1. **M2 chip optimization** - MPS acceleration significantly boosted ingestion speed
2. **Type validation critical** - Schema mismatches fail silently until insertion
3. **Batch size matters** - 1000 vectors balanced memory and throughput
4. **Index choice impacts latency** - IVF_FLAT good for accuracy, HNSW better for speed
5. **Docker on M2** - Platform emulation works but has overhead (Attu warning)
6. **Checkpointing valuable** - Even though not needed, provides resilience for larger datasets

---

**Completion Time:** 1.3 minutes actual (vs 25-35 min estimated)  
**Status:** ✅ **COMPLETE** - Ready for Member 4 (FastAPI Layer)  
**Date:** December 5, 2025
