"""
Test and validate Milvus collection
Run sample queries to verify ingestion and search functionality
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent))
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, SEARCH_PARAMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MilvusQueryTester:
    """Test Milvus collection with sample queries"""
    
    def __init__(self):
        self.collection = None
        self.model = None
        
    def connect(self):
        """Connect to Milvus"""
        logger.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("✓ Connected")
        
    def load_collection(self):
        """Load collection"""
        logger.info(f"Loading collection: {COLLECTION_NAME}")
        self.collection = Collection(COLLECTION_NAME)
        
        # Ensure collection is loaded in memory
        self.collection.load()
        
        # Get stats
        num_entities = self.collection.num_entities
        logger.info(f"✓ Collection loaded: {num_entities:,} entities")
        
    def load_model(self):
        """Load sentence transformer model"""
        logger.info("Loading sentence-transformers model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✓ Model loaded")
        
    def query_text(self, query: str, top_k: int = 5, index_type: str = "IVF_FLAT") -> List[Dict]:
        """
        Query with natural language text
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            index_type: Index type for search params
            
        Returns:
            List of result dictionaries
        """
        # Generate embedding for query
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Search parameters
        search_params = SEARCH_PARAMS[index_type]
        
        # Execute search
        start_time = time.time()
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["paper_id", "title", "abstract", "categories", "text_length"]
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "paper_id": hit.entity.get("paper_id"),
                    "title": hit.entity.get("title"),
                    "abstract": hit.entity.get("abstract")[:200] + "...",  # Truncate
                    "categories": hit.entity.get("categories"),
                    "score": hit.score,
                    "distance": hit.distance
                })
        
        return formatted_results, elapsed_ms
    
    def run_test_queries(self):
        """Run a set of test queries"""
        test_queries = [
            "neural networks deep learning",
            "quantum computing algorithms",
            "natural language processing transformers",
            "computer vision image recognition",
            "reinforcement learning robotics",
            "graph neural networks",
            "time series forecasting",
            "recommendation systems collaborative filtering"
        ]
        
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING TEST QUERIES")
        logger.info(f"{'='*80}\n")
        
        total_time = 0
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nQuery {i}: \"{query}\"")
            logger.info("-" * 80)
            
            results, elapsed_ms = self.query_text(query, top_k=3)
            total_time += elapsed_ms
            
            logger.info(f"Search time: {elapsed_ms:.2f}ms")
            logger.info(f"\nTop 3 Results:")
            
            for j, result in enumerate(results, 1):
                logger.info(f"\n  {j}. Score: {result['score']:.4f}")
                logger.info(f"     Paper: {result['paper_id']}")
                logger.info(f"     Title: {result['title']}")
                logger.info(f"     Categories: {result['categories']}")
                logger.info(f"     Abstract: {result['abstract']}")
        
        avg_time = total_time / len(test_queries)
        logger.info(f"\n{'='*80}")
        logger.info(f"QUERY PERFORMANCE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total queries: {len(test_queries)}")
        logger.info(f"Average latency: {avg_time:.2f}ms")
        logger.info(f"95th percentile target: <120ms")
        logger.info(f"Status: {'✓ PASS' if avg_time < 120 else '✗ NEEDS OPTIMIZATION'}")
        logger.info(f"{'='*80}\n")
    
    def validate_collection(self):
        """Validate collection properties"""
        logger.info(f"\n{'='*80}")
        logger.info("COLLECTION VALIDATION")
        logger.info(f"{'='*80}")
        
        # Check entity count
        num_entities = self.collection.num_entities
        expected_entities = 510203  # From Member 2 output
        
        logger.info(f"Entity count: {num_entities:,}")
        logger.info(f"Expected count: {expected_entities:,}")
        
        if num_entities == expected_entities:
            logger.info("✓ Entity count matches")
        else:
            logger.warning(f"⚠ Entity count mismatch (difference: {abs(num_entities - expected_entities):,})")
        
        # Check indexes
        indexes = self.collection.indexes
        logger.info(f"\nIndexes: {len(indexes)}")
        for idx in indexes:
            logger.info(f"  - Field: {idx.field_name}, Type: {idx.params.get('index_type', 'N/A')}")
        
        if indexes:
            logger.info("✓ Index present")
        else:
            logger.warning("⚠ No index found")
        
        # Check schema
        schema = self.collection.schema
        logger.info(f"\nSchema fields: {len(schema.fields)}")
        for field in schema.fields:
            logger.info(f"  - {field.name}: {field.dtype}")
        
        logger.info(f"{'='*80}\n")
    
    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        logger.info("✓ Disconnected from Milvus")


def main():
    """Main test function"""
    tester = MilvusQueryTester()
    
    try:
        # Connect and load
        tester.connect()
        tester.load_collection()
        tester.load_model()
        
        # Validate collection
        tester.validate_collection()
        
        # Run test queries
        tester.run_test_queries()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise
    finally:
        tester.disconnect()


if __name__ == "__main__":
    main()
