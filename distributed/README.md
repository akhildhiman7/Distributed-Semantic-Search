# Distributed Semantic Search Architecture

## Overview

This directory contains the configuration and scripts to transform the standalone semantic search system into a **true distributed system** with:

- **Distributed Milvus Cluster** (14 containers)
- **Load-Balanced API Layer** (3 instances + NGINX)
- **Data Sharding** across multiple nodes
- **Replication** for fault tolerance
- **Parallel Query Execution**

---

## Architecture Comparison

### Current: Standalone (Single-Node)

```
┌─────────────────────────────────────────┐
│           Client Requests               │
└─────────────────┬───────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Single API Instance │
       │   (localhost:8000)   │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Milvus Standalone   │
       │   (all-in-one)       │
       └──────────────────────┘
```

**Limitations:**

- ❌ Single point of failure
- ❌ No horizontal scaling
- ❌ Limited throughput
- ❌ No fault tolerance
- ❌ Vertical scaling only

---

### Distributed: Multi-Node Cluster

```
┌─────────────────────────────────────────────────────┐
│              Client Requests                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   NGINX Load Balancer│
        │   (localhost:8000)   │
        └─┬────────┬──────────┬─┘
          │        │          │
          ▼        ▼          ▼
     ┌─────────────────────────────────┐
     │   API Instances (3 replicas)    │
     │   api-1, api-2, api-3           │
     └─┬──────────┬───────────┬────────┘
       │          │           │
       └──────────┴───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Milvus Proxy (2)   │
        │  Load balanced      │
        └─────┬──────┬────────┘
              │      │
    ┌─────────┴──────┴──────────┐
    │                            │
    ▼                            ▼
┌─────────────────────┐  ┌─────────────────────┐
│  Milvus Cluster     │  │  Milvus Cluster     │
│  (14 components)    │  │  Components:        │
│                     │  │                     │
│  - RootCoord (1)    │  │  - QueryNode (3)    │
│  - DataCoord (1)    │  │  - DataNode (2)     │
│  - QueryCoord (1)   │  │  - IndexNode (2)    │
│  - IndexCoord (1)   │  │                     │
└─────────────────────┘  └─────────────────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Distributed  │
              │  Storage      │
              │  (Sharded)    │
              └───────────────┘
```

**Benefits:**

- ✅ High availability (replicas survive failures)
- ✅ Horizontal scaling (add more nodes)
- ✅ Higher throughput (parallel processing)
- ✅ Load balancing (requests distributed)
- ✅ Data sharding (parallelized search)

---

## Components Breakdown

### 1. Milvus Distributed Cluster (14 containers)

| Component      | Count | Purpose                                | Port         |
| -------------- | ----- | -------------------------------------- | ------------ |
| **RootCoord**  | 1     | Cluster coordinator, manages metadata  | -            |
| **QueryCoord** | 1     | Query task scheduling & load balancing | -            |
| **DataCoord**  | 1     | Data segment management                | -            |
| **IndexCoord** | 1     | Index building coordination            | -            |
| **Proxy**      | 2     | Entry points (load balanced)           | 19531, 19532 |
| **QueryNode**  | 3     | Query execution (replicas)             | -            |
| **DataNode**   | 2     | Data storage & persistence             | -            |
| **IndexNode**  | 2     | Index building                         | -            |
| **Etcd**       | 1     | Metadata storage                       | 2379         |
| **MinIO**      | 1     | Object storage                         | 9000-9001    |
| **Pulsar**     | 1     | Message queue                          | 6650         |
| **Attu**       | 1     | Web UI                                 | 3002         |

### 2. API Layer (4 containers)

| Component         | Count | Purpose         | Port            |
| ----------------- | ----- | --------------- | --------------- |
| **NGINX**         | 1     | Load balancer   | 8000 (external) |
| **API Instances** | 3     | FastAPI servers | Internal        |

---

## Key Features

### Data Sharding

```python
# Collection created with 4 shards
COLLECTION_CONFIG = {
    "shards_num": 4,  # Data distributed across 4 shards
}

# Benefits:
# - 510K vectors split into 4 partitions (~127K each)
# - Parallel search across shards
# - Better memory distribution
```

**How it works:**

```
Total Data: 510,203 vectors (384-dim)
    │
    ├─→ Shard 1: ~127,050 vectors (DataNode-1)
    ├─→ Shard 2: ~127,050 vectors (DataNode-1)
    ├─→ Shard 3: ~127,050 vectors (DataNode-2)
    └─→ Shard 4: ~127,053 vectors (DataNode-2)

Search Query:
    └─→ Broadcast to all 4 shards (parallel)
        ├─→ QueryNode-1: searches Shard 1 & 2
        ├─→ QueryNode-2: searches Shard 3
        └─→ QueryNode-3: searches Shard 4
    └─→ Merge results (top-K from each shard)
```

### Replication

```python
# 3 replicas for fault tolerance
REPLICA_CONFIG = {
    "replica_number": 3,  # Each shard replicated 3 times
}
```

**How it works:**

```
Shard 1:
    ├─→ Replica A (QueryNode-1) - Primary
    ├─→ Replica B (QueryNode-2) - Standby
    └─→ Replica C (QueryNode-3) - Standby

If QueryNode-1 fails:
    └─→ QueryCoord routes queries to QueryNode-2
    └─→ No data loss, automatic failover
```

### Load Balancing

**NGINX Level:**

```nginx
upstream api_backend {
    least_conn;  # Route to least busy instance
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

**Milvus Level:**

```python
# 2 proxies for client-side load balancing
MILVUS_PROXIES = [
    {"host": "localhost", "port": 19531},  # proxy-1
    {"host": "localhost", "port": 19532},  # proxy-2
]
```

---

## Setup Instructions

### Prerequisites

- Docker Desktop with **16GB+ RAM** allocated
- Docker Compose v2.0+
- 50GB+ free disk space
- Existing data ingested (510K vectors)

### Step 1: Stop Standalone System

```bash
cd milvus
docker compose down

cd ../api
# Stop API if running
```

### Step 2: Start Distributed Milvus Cluster

```bash
cd distributed
docker compose -f docker-compose-cluster.yml up -d

# Wait for all services to be healthy (~2 minutes)
docker ps --filter "name=milvus" --format "table {{.Names}}\t{{.Status}}"
```

**Expected output:**

```
NAMES                   STATUS
milvus-attu            Up (healthy)
milvus-proxy-1         Up
milvus-proxy-2         Up
milvus-querynode-1     Up
milvus-querynode-2     Up
milvus-querynode-3     Up
milvus-datanode-1      Up
milvus-datanode-2      Up
milvus-indexnode-1     Up
milvus-indexnode-2     Up
milvus-rootcoord       Up
milvus-datacoord       Up
milvus-querycoord      Up
milvus-indexcoord      Up
```

### Step 3: Setup Collection with Sharding

```bash
cd distributed
python setup_cluster.py
```

This will:

- Create collection with 4 shards
- Configure 3 replicas
- Create HNSW index
- Verify distribution

### Step 4: Ingest Data (Optional - if starting fresh)

```bash
# Use existing data
cd ../milvus
python ingest.py --collection arxiv_papers_distributed
```

### Step 5: Start Distributed API Layer

```bash
cd ../distributed
docker compose -f docker-compose-api.yml up -d
```

**Expected output:**

```
NAMES                   STATUS              PORTS
api-load-balancer      Up (healthy)        0.0.0.0:8000->80/tcp
api-instance-1         Up (healthy)
api-instance-2         Up (healthy)
api-instance-3         Up (healthy)
```

### Step 6: Verify System

```bash
# Test load balancer
curl http://localhost:8000/health

# Check which instance served the request
curl -i http://localhost:8000/health | grep X-Upstream-Server

# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'
```

---

## Performance Testing

### Run Distributed Load Test

```bash
cd distributed
python test_distributed.py
```

**Expected improvements:**

- **Throughput**: 2-3x higher RPS
- **Latency**: 30-50% lower p95 latency
- **Fault Tolerance**: Survives node failures
- **Scalability**: Linear scaling with nodes

### Benchmark Results (Expected)

| Metric             | Standalone | Distributed | Improvement |
| ------------------ | ---------- | ----------- | ----------- |
| **RPS**            | 39         | 100-120     | +157-207%   |
| **p50 Latency**    | 152ms      | 80-100ms    | -34-47%     |
| **p95 Latency**    | 465ms      | 200-250ms   | -46-57%     |
| **p99 Latency**    | 2812ms     | 500-800ms   | -72-82%     |
| **Max Concurrent** | ~50        | 200+        | +300%       |

---

## Monitoring

### Access Points

- **Attu (Milvus UI)**: http://localhost:3002
- **NGINX Stats**: http://localhost:8080/nginx_status
- **Grafana**: http://localhost:3001 (if monitoring stack running)
- **Prometheus**: http://localhost:9090

### Key Metrics to Monitor

**Milvus Cluster:**

```promql
# Query distribution across nodes
milvus_querynode_search_qps

# Shard balance
milvus_datanode_segment_num

# Replica health
milvus_querycoord_replica_num
```

**API Layer:**

```promql
# Requests per instance
rate(search_requests_total[5m])

# Load balance distribution
nginx_upstream_requests_total by (instance)
```

---

## Scaling Guide

### Horizontal Scaling

**Add More Query Nodes:**

```yaml
# In docker-compose-cluster.yml
querynode-4:
  container_name: milvus-querynode-4
  image: milvusdb/milvus:v2.3.3
  command: ["milvus", "run", "querynode"]
  # ... same config as other querynodes
```

**Add More API Instances:**

```yaml
# In docker-compose-api.yml
api-4:
  container_name: api-instance-4
  # ... same config as other API instances

# Update nginx.conf
upstream api_backend {
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
    server api-4:8000;  # Add new instance
}
```

### Increase Replicas

```python
# Increase replica count for higher availability
REPLICA_CONFIG = {
    "replica_number": 4,  # 4 replicas (requires 4+ query nodes)
}

# Reload collection to apply
collection.load(replica_number=4)
```

---

## Troubleshooting

### Cluster Not Starting

**Issue:** Some containers fail to start

**Solution:**

```bash
# Check logs
docker logs milvus-rootcoord
docker logs milvus-proxy-1

# Common issues:
# 1. Insufficient memory - allocate 16GB+ to Docker
# 2. Pulsar not ready - wait 60s for all services
# 3. Etcd connection issues - check network
```

### Load Balancer Returns 502

**Issue:** NGINX returns "Bad Gateway"

**Solution:**

```bash
# Check API instances health
docker ps --filter "name=api-instance"

# Restart unhealthy instances
docker restart api-instance-1

# Check nginx logs
docker logs api-load-balancer
```

### Replicas Not Distributing

**Issue:** All queries go to one query node

**Solution:**

```python
# Check replica distribution
collection.get_replicas()

# Reload with explicit replica placement
collection.load(replica_number=3)
```

---

## Cost Comparison

### Resource Usage

| Setup           | Containers | RAM      | CPU        | Disk  |
| --------------- | ---------- | -------- | ---------- | ----- |
| **Standalone**  | 4          | 4-6 GB   | 2-4 cores  | 10 GB |
| **Distributed** | 18         | 16-24 GB | 8-16 cores | 25 GB |

### When to Use Each

**Standalone:**

- ✅ Development/testing
- ✅ Small datasets (<1M vectors)
- ✅ Low query volume (<10 QPS)
- ✅ Limited resources
- ✅ Single-tenant applications

**Distributed:**

- ✅ Production deployments
- ✅ Large datasets (>1M vectors)
- ✅ High query volume (>50 QPS)
- ✅ Multi-tenant applications
- ✅ Fault tolerance required
- ✅ Scaling anticipated

---

## Next Steps

1. **Optimize Index**: Test HNSW vs IVF_FLAT performance
2. **Tune Sharding**: Experiment with 2, 4, 8, 16 shards
3. **Add Caching**: Redis for frequent queries
4. **Multi-Region**: Deploy across availability zones
5. **Auto-Scaling**: Implement horizontal pod autoscaling

---

## Files in This Directory

```
distributed/
├── docker-compose-cluster.yml  # Milvus 14-container cluster
├── docker-compose-api.yml      # 3 API instances + NGINX
├── nginx.conf                  # Load balancer configuration
├── distributed_config.py       # Sharding & replication config
├── setup_cluster.py           # Setup script with verification
├── test_distributed.py        # Performance comparison tests
└── README.md                  # This file
```

---

## References

- [Milvus Distributed Architecture](https://milvus.io/docs/architecture_overview.md)
- [Data Sharding Guide](https://milvus.io/docs/sharding.md)
- [Replica Management](https://milvus.io/docs/replica.md)
- [Performance Tuning](https://milvus.io/docs/performance_faq.md)

---

**Status**: 🚀 Ready for deployment (requires 16GB+ RAM)
