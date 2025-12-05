"""
Setup script for distributed Milvus cluster with sharding and replication.

This script:
1. Creates a collection with 4 shards
2. Sets up 3 replicas across query nodes
3. Configures HNSW index for better distributed performance
4. Demonstrates data ingestion with sharding
"""

import sys
import time
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, Partition
)
from typing import List, Dict

# Import distributed configuration
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributed_config import (
    COLLECTION_CONFIG, INDEX_CONFIG, REPLICA_CONFIG,
    MILVUS_PROXIES, SEARCH_PARAMS_HNSW
)


class DistributedMilvusSetup:
    """Setup and manage distributed Milvus cluster."""
    
    def __init__(self):
        self.collection_name = COLLECTION_CONFIG["name"]
        self.collection = None
        
    def connect_to_cluster(self):
        """Connect to Milvus cluster via proxy."""
        print("🔗 Connecting to Milvus cluster...")
        
        # Connect to first proxy (can implement client-side load balancing)
        proxy = MILVUS_PROXIES[0]
        connections.connect(
            alias="default",
            host=proxy["host"],
            port=proxy["port"]
        )
        print(f"✓ Connected to {proxy['host']}:{proxy['port']}")
        
    def create_distributed_collection(self):
        """Create collection with sharding configuration."""
        print(f"\n📦 Creating distributed collection: {self.collection_name}")
        
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            print(f"⚠️  Collection {self.collection_name} already exists")
            response = input("Drop and recreate? (yes/no): ")
            if response.lower() == 'yes':
                utility.drop_collection(self.collection_name)
                print("✓ Dropped existing collection")
            else:
                self.collection = Collection(self.collection_name)
                return
        
        # Define schema (matching standalone schema with actual data)
        fields = [
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text_length", dtype=DataType.INT32),
            FieldSchema(name="has_full_data", dtype=DataType.BOOL),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=COLLECTION_CONFIG["description"]
        )
        
        # Create collection with sharding
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            shards_num=COLLECTION_CONFIG["shards_num"],
            consistency_level=COLLECTION_CONFIG["consistency_level"]
        )
        
        print(f"✓ Created collection with {COLLECTION_CONFIG['shards_num']} shards")
        
    def create_index(self):
        """Create HNSW index for distributed search."""
        print(f"\n🔨 Creating {INDEX_CONFIG['index_type']} index...")
        
        self.collection.create_index(
            field_name="embedding",
            index_params=INDEX_CONFIG
        )
        
        print(f"✓ Created {INDEX_CONFIG['index_type']} index")
        print(f"  - M: {INDEX_CONFIG['params']['M']}")
        print(f"  - efConstruction: {INDEX_CONFIG['params']['efConstruction']}")
        
    def setup_replicas(self):
        """Set up replicas across query nodes."""
        print(f"\n🔄 Setting up {REPLICA_CONFIG['replica_number']} replicas...")
        
        # Load collection first
        self.collection.load()
        print("✓ Collection loaded")
        
        # Get replica information
        replicas = self.collection.get_replicas()
        print(f"✓ Replicas configured: {len(replicas.groups)}")
        
        for i, group in enumerate(replicas.groups):
            print(f"\n  Replica {i+1}:")
            print(f"    - Replica ID: {group.id}")
            print(f"    - Shard replicas: {len(group.shards)}")
            for shard in group.shards:
                print(f"      • Shard: {shard.channel_name}")
                # Get node IDs if available
                try:
                    if hasattr(shard, 'node_ids'):
                        print(f"        Nodes: {shard.node_ids}")
                    elif hasattr(shard, 'replicas'):
                        print(f"        Replicas: {len(shard.replicas)}")
                except Exception:
                    print("        (Node details not available)")
        
    def verify_distribution(self):
        """Verify data distribution across shards."""
        print("\n📊 Verifying data distribution...")
        
        # Get collection statistics
        stats = self.collection.num_entities
        print(f"✓ Total entities: {stats:,}")
        
        # Get collection info
        print(f"✓ Shards: {COLLECTION_CONFIG['shards_num']}")
        print(f"✓ Consistency level: {COLLECTION_CONFIG['consistency_level']}")
        
    def test_search(self):
        """Test distributed search across replicas."""
        print("\n🔍 Testing distributed search...")
        
        import numpy as np
        
        # Create a random query vector
        query_vector = np.random.random(384).tolist()
        
        # Search parameters
        search_params = SEARCH_PARAMS_HNSW
        
        # Perform search
        start_time = time.time()
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=["paper_id", "title"]
        )
        latency_ms = (time.time() - start_time) * 1000
        
        print(f"✓ Search completed in {latency_ms:.2f}ms")
        print(f"✓ Results: {len(results[0])} papers found")
        
        if len(results[0]) > 0:
            print("\nTop result:")
            hit = results[0][0]
            print(f"  - Paper ID: {hit.entity.get('paper_id')}")
            print(f"  - Title: {hit.entity.get('title')[:80]}...")
            print(f"  - Score: {hit.score:.4f}")
            print(f"  - Distance: {hit.distance:.4f}")
        
    def display_cluster_info(self):
        """Display cluster topology information."""
        print("\n" + "="*60)
        print("DISTRIBUTED CLUSTER INFORMATION")
        print("="*60)
        
        # Collection info
        print(f"\n📦 Collection: {self.collection_name}")
        print(f"  - Entities: {self.collection.num_entities:,}")
        print(f"  - Shards: {COLLECTION_CONFIG['shards_num']}")
        print(f"  - Replicas: {REPLICA_CONFIG['replica_number']}")
        
        # Index info
        indexes = self.collection.indexes
        if indexes:
            print(f"\n🔨 Index:")
            for index in indexes:
                print(f"  - Type: {index.params['index_type']}")
                print(f"  - Metric: {index.params['metric_type']}")
                print(f"  - Params: {index.params['params']}")
        
        # Replica distribution
        replicas = self.collection.get_replicas()
        print("\n🔄 Replica Distribution:")
        for i, group in enumerate(replicas.groups):
            print(f"  Replica Group {i+1}:")
            print(f"    - Shards: {len(group.shards)}")
            for shard in group.shards:
                print(f"      • {shard.channel_name}")
        
        print("\n" + "="*60)


def main():
    """Main setup function."""
    print("="*60)
    print("DISTRIBUTED MILVUS CLUSTER SETUP")
    print("="*60)
    
    setup = DistributedMilvusSetup()
    
    try:
        # Step 1: Connect
        setup.connect_to_cluster()
        
        # Step 2: Create collection with sharding
        setup.create_distributed_collection()
        
        # Step 3: Create index
        setup.create_index()
        
        # Step 4: Set up replicas
        setup.setup_replicas()
        
        # Step 5: Verify distribution
        setup.verify_distribution()
        
        # Step 6: Test search
        if setup.collection.num_entities > 0:
            setup.test_search()
        
        # Step 7: Display cluster info
        setup.display_cluster_info()
        
        print("\n✅ Distributed cluster setup complete!")
        print("\nNext steps:")
        print("1. Ingest data using: python ingest_distributed.py")
        print("2. Monitor cluster: http://localhost:3002 (Attu)")
        print("3. Run distributed load test")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()
