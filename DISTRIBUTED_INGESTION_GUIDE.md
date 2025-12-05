# Distributed Milvus Ingestion Guide

## Quick Start

### 1. Ensure Cluster is Running

```bash
cd distributed
docker compose up -d
```

Wait 30 seconds for all services to initialize.

### 2. Verify Cluster Health

```bash
# Check all containers are up
docker ps --filter name=milvus | wc -l
# Should show 19 (18 containers + header)

# Test connection
cd ..
source .venv/bin/activate
python -c "from pymilvus import connections, utility; connections.connect('default', host='localhost', port=19531); print('Collections:', utility.list_collections())"
```

### 3. Create Collection (if not exists)

```bash
cd distributed
source ../.venv/bin/activate
python setup_cluster.py
# Type 'yes' when prompted
```

### 4. Run Data Ingestion

**Option A: Background (recommended for long-running)**

```bash
cd ..
source .venv/bin/activate
nohup python distributed/ingest_distributed.py > distributed_ingest.log 2>&1 &
```

**Option B: Foreground (see progress)**

```bash
cd ..
source .venv/bin/activate
python distributed/ingest_distributed.py
```

### 5. Monitor Progress

**Check log file (Option A):**

```bash
tail -f distributed_ingest.log
```

**Check entity count:**

```bash
python -c "from pymilvus import connections, Collection; connections.connect('default', host='localhost', port=19531); c = Collection('arxiv_papers_distributed'); print(f'Entities: {c.num_entities:,}')"
```

**Check process status:**

```bash
ps aux | grep ingest_distributed
```

---

## Expected Performance

⚠️ **WARNING:** Distributed ingestion is **extremely slow** on 8GB RAM M2 Mac:

- **Standalone:** 1000 vectors in 2-3 seconds
- **Distributed:** 1000 vectors in 3-4 minutes (**100x slower!**)

**Why so slow:**

- 18-component coordination overhead
- Kafka message queue latency
- Memory pressure causing swap usage
- gRPC serialization between containers

**Time estimate for 510K vectors:**

- At 3 minutes per 1000 vectors
- 510 batches × 3 minutes = **~25 hours!**

---

## Recommended Workflow

### For Testing (Small Dataset)

Test with first 10K vectors only:

```python
# Edit distributed/ingest_distributed.py
# Add at line ~55 after loading embeddings:

embeddings = embeddings[:10000]  # Test with 10K only
metadata = metadata.iloc[:10000]
```

Then run ingestion (~30 minutes for 10K vectors).

### For Production (Full Dataset)

**Option 1:** Use standalone Milvus instead

- Already ingested: 510K vectors
- Performance: 39 RPS, <200ms latency
- Proven stable

```bash
cd milvus
docker compose up -d
```

**Option 2:** Wait for better hardware

- Need 16GB+ RAM for acceptable distributed performance
- Or deploy on cloud instance (AWS/GCP with proper resources)

---

## Troubleshooting

### Ingestion Hangs or Times Out

**Symptoms:**

- Progress bar stuck at 0%
- No log output for minutes
- insert() or load() hangs

**Solutions:**

1. **Restart cluster** (clears any stuck states)

```bash
cd distributed
docker compose down
docker compose up -d
sleep 30  # Wait for initialization
```

2. **Reduce batch size** (in `distributed/ingest_distributed.py`)

```python
# Line ~18
BATCH_SIZE = 100  # Default is 1000, try 100
```

3. **Check container logs**

```bash
docker logs milvus-rootcoord 2>&1 | tail -50
docker logs milvus-datanode-1 2>&1 | tail -50
docker logs milvus-kafka 2>&1 | tail -50
```

### Out of Memory Errors

**Symptoms:**

- Containers restart frequently
- "OOMKilled" in docker ps
- System becomes unresponsive

**Solutions:**

1. **Increase memory limits** (in `distributed/docker-compose-cluster.yml`)

```yaml
querynode-1:
  mem_limit: 2g # Increase from 1g
datanode-1:
  mem_limit: 3g # Increase from 2g
```

2. **Close other applications** to free RAM

3. **Use standalone instead** - distributed needs 16GB+ RAM

### Collection Not Found

**Symptoms:**

```
Collection 'arxiv_papers_distributed' not exist
```

**Solution:**

```bash
cd distributed
source ../.venv/bin/activate
python setup_cluster.py
```

---

## Performance Monitoring

### Check Ingestion Rate

```bash
# Before
python -c "from pymilvus import connections, Collection; connections.connect('default', host='localhost', port=19531); c = Collection('arxiv_papers_distributed'); print(f'Before: {c.num_entities:,}')"

# Wait 5 minutes

# After
python -c "from pymilvus import connections, Collection; connections.connect('default', host='localhost', port=19531); c = Collection('arxiv_papers_distributed'); print(f'After: {c.num_entities:,}')"

# Calculate rate: (After - Before) / 300 seconds = vectors/sec
```

### Monitor Resource Usage

```bash
# Real-time container stats
docker stats

# Memory usage summary
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}" | grep milvus
```

---

## Kill/Restart Ingestion

### Stop Background Process

```bash
# Find process ID
ps aux | grep ingest_distributed

# Kill it
kill <PID>
# Or force kill
kill -9 <PID>
```

### Restart from Scratch

```bash
# Stop ingestion
kill $(ps aux | grep ingest_distributed | grep -v grep | awk '{print $2}')

# Drop and recreate collection
cd distributed
source ../.venv/bin/activate
python -c "from pymilvus import connections, utility; connections.connect('default', host='localhost', port=19531); utility.drop_collection('arxiv_papers_distributed')"
python setup_cluster.py

# Start fresh ingestion
cd ..
nohup python distributed/ingest_distributed.py > distributed_ingest.log 2>&1 &
```

---

## What to Tell ChatGPT

If asking ChatGPT for help optimizing this:

**Copy/paste this context:**

> I have a distributed Milvus v2.4.8 cluster (18 containers: Kafka, Zookeeper, 4 coordinators, 3 query nodes, 2 data nodes, 2 index nodes, 2 proxies, etcd, MinIO, Attu) running on M2 Mac with 8GB RAM.
>
> All containers are healthy and I can connect/create collections, but ingestion is extremely slow: 1000 vectors takes 3-4 minutes (vs 2-3 seconds on standalone).
>
> **Cluster config:**
>
> - Memory limits: 1GB per query node, 2GB per data/index node
> - mmap enabled on all nodes
> - Kafka: 340MB RAM
> - 4 shards, replication factor 3
> - Collection: 7 fields (paper_id, embedding[384], title, abstract, categories, text_length, has_full_data)
> - Dataset: 510K vectors total
>
> **Symptoms:**
>
> - First batch insert takes 3-4 minutes
> - Operations timeout frequently
> - Datanode logs show "channel not found" warnings
> - Heavy coordination overhead between components
>
> **Goal:** Either optimize for faster ingestion OR confirm that 8GB RAM is insufficient and recommend minimum specs.
>
> Docker compose: `/distributed/docker-compose-cluster.yml`
> Ingestion script: `/distributed/ingest_distributed.py`
> Collection setup: `/distributed/setup_cluster.py`

---

## File Locations

- **Cluster config:** `distributed/docker-compose-cluster.yml`
- **Milvus config:** `distributed/milvus.yaml`
- **Collection setup:** `distributed/setup_cluster.py`
- **Ingestion script:** `distributed/ingest_distributed.py`
- **Performance test:** `distributed/test_distributed.py`
- **Embeddings:** `embed/embeddings/full/*.npy` (9 files)
- **Metadata:** `data/partitions/*/part-*.json`
- **Logs:** Check `distributed_ingest.log` or docker logs

---

## Bottom Line

**Distributed Milvus on 8GB RAM M2 Mac:**

- ✅ **Starts and runs** - technically functional
- ❌ **Too slow for real use** - 100x slower than standalone
- 💡 **Recommendation:** Use standalone (39 RPS, <200ms latency, proven stable)

**If you MUST use distributed:**

- Get 16GB+ RAM hardware
- Or deploy on cloud with proper resources
- Or be prepared to wait 20-30 hours for full ingestion
