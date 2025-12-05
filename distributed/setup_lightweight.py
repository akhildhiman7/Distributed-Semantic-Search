"""
Setup script for lightweight distributed cluster (8GB RAM optimized).

This creates:
- 2 shards for data distribution
- 2 replicas for fault tolerance
- HNSW index for better performance
"""

import sys
import time
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility
)


def setup_lightweight_cluster():
    """Setup lightweight distributed cluster."""
    
    print("="*60)
    print("LIGHTWEIGHT DISTRIBUTED CLUSTER SETUP")
    print("="*60)
    
    # Connect to cluster
    print("\n🔗 Connecting to Milvus cluster...")
    connections.connect(
        alias="default",
        host="localhost",
        port=19530
    )
    print("✓ Connected")
    
    # Collection configuration
    collection_name = "arxiv_papers"
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        print(f"\n📦 Collection '{collection_name}' already exists")
        print("Using existing collection...")
        collection = Collection(collection_name)
    else:
        print(f"\n📦 Creating collection '{collection_name}'...")
        
        # Define schema (same as original)
        fields = [
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="ArXiv papers with 2 shards"
        )
        
        # Create with 2 shards (distributed across query nodes)
        collection = Collection(
            name=collection_name,
            schema=schema,
            shards_num=2,  # 2 shards for distribution
            consistency_level="Strong"
        )
        print("✓ Created collection with 2 shards")
    
    # Check if index exists
    indexes = collection.indexes
    if indexes:
        print(f"\n🔨 Index already exists: {indexes[0].params['index_type']}")
    else:
        print("\n🔨 Creating HNSW index (optimized for 8GB RAM)...")
        
        # HNSW index with reduced parameters for 8GB RAM
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 8,              # Reduced from 16 (saves memory)
                "efConstruction": 100 # Reduced from 200 (saves memory)
            }
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("✓ Created HNSW index")
        print("  - M: 8 (memory optimized)")
        print("  - efConstruction: 100")
    
    # Load collection with replicas
    print("\n🔄 Loading collection with 2 replicas...")
    collection.load(replica_number=2)
    
    # Wait for loading
    print("⏳ Waiting for replicas to be ready...")
    time.sleep(3)
    
    # Verify setup
    print("\n📊 Verifying cluster setup...")
    
    # Get replica information
    replicas = collection.get_replicas()
    print(f"✓ Replicas: {len(replicas.groups)}")
    
    for i, group in enumerate(replicas.groups):
        print(f"\n  Replica Group {i+1}:")
        print(f"    - Replica ID: {group.id}")
        print(f"    - Shard replicas: {len(group.shards)}")
        for shard in group.shards:
            print(f"      • {shard.channel_name}")
            print(f"        Query nodes: {shard.node_ids}")
    
    # Collection stats
    stats = collection.num_entities
    print(f"\n✓ Total entities: {stats:,}")
    
    # Index info
    if collection.indexes:
        index = collection.indexes[0]
        print(f"\n✓ Index type: {index.params['index_type']}")
        print(f"✓ Metric type: {index.params['metric_type']}")
    
    # Test search
    if stats > 0:
        print("\n🔍 Testing distributed search...")
        import numpy as np
        
        query_vector = np.random.random(384).tolist()
        
        start = time.time()
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 32}},
            limit=10,
            output_fields=["paper_id", "title"]
        )
        latency_ms = (time.time() - start) * 1000
        
        print(f"✓ Search completed in {latency_ms:.2f}ms")
        print(f"✓ Found {len(results[0])} results")
        
        if len(results[0]) > 0:
            hit = results[0][0]
            print(f"\nTop result:")
            print(f"  - Paper ID: {hit.entity.get('paper_id')}")
            print(f"  - Score: {hit.score:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("CLUSTER CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Architecture: Lightweight Distributed")
    print(f"Shards: 2 (data distributed)")
    print(f"Replicas: 2 (fault tolerant)")
    print(f"Query Nodes: 2 (parallel search)")
    print(f"Index Type: HNSW (fast)")
    print(f"Total Entities: {stats:,}")
    print(f"Memory Usage: ~6GB")
    print("="*60)
    
    print("\n✅ Lightweight distributed cluster ready!")
    print("\nNext steps:")
    print("1. If entities = 0, run: python milvus/ingest.py")
    print("2. Test performance: python monitoring/loadtest/load_test.py --concurrency 20 --duration 30")
    print("3. Monitor: docker stats")
    
    connections.disconnect("default")


if __name__ == "__main__":
    try:
        setup_lightweight_cluster()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
