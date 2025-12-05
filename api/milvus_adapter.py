from pymilvus import connections, Collection, utility
from typing import List, Optional, Dict, Any
import time

from config import MilvusConfig
from api.dependencies import normalize_categories

class MilvusAdapter:
    def __init__(self, config: MilvusConfig = None):
        self.config = config or MilvusConfig()
        self._connect()
        self.collection = self._get_collection()
    
    def _connect(self):
        """Establish connection to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=int(self.config.port),
                timeout=30
            )
            print(f"✅ Connected to Milvus at {self.config.host}:{self.config.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _get_collection(self) -> Collection:
            """Get or create collection"""
            collection_name = self.config.collection_name
            
            # Check if collection exists
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist. "
                            f"Run create_collection.py first.")
            
            collection = Collection(collection_name)
            
            # Load collection - older pymilvus versions don't have is_loaded
            try:
                # Try to load - if already loaded, this might fail or be ignored
                print(f"Loading collection '{collection_name}'...")
                collection.load()
                print(f"✅ Collection loaded with {collection.num_entities} entities")
            except Exception as e:
                # If load fails, it might already be loaded or have issues
                print(f"⚠️  Note: Collection load returned: {e}")
                print(f"   Collection has {collection.num_entities} entities")
            
            return collection
    
    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        categories: Optional[List[str]] = None,
        expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar papers
        
        Args:
            vector: Query embedding vector
            top_k: Number of results
            categories: Filter by categories
            expr: Additional filter expression
        
        Returns:
            List of matching papers with scores
        """
        # Build search expression
        search_expr = expr or ""
        
        if categories:
            categories_str = normalize_categories(categories)
            if categories_str:
                category_expr = f'categories like "%{categories_str}%"'
                search_expr = category_expr if not search_expr else f"{search_expr} and {category_expr}"
        
        # Search parameters (matching his HNSW index)
        search_params = {
            "metric_type": "IP",  # Inner Product (since embeddings are normalized)
            "params": {"ef": 100}  # HNSW search parameter
        }
        
        # Perform search
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=search_expr if search_expr else None,
            output_fields=["paper_id", "title", "abstract", "categories"]
        )
        
        # Format results
        output = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                output.append({
                    "paper_id": entity.get("paper_id"),
                    "title": entity.get("title"),
                    "abstract": entity.get("abstract"),
                    "categories": entity.get("categories", "").split(",") if entity.get("categories") else [],
                    "score": float(hit.score)  # Note: score for IP, not distance
                })
        
        return output
    
    def insert_paper(
        self,
        paper_id: int,
        title: str,
        abstract: str,
        categories: List[str],
        vector: List[float]
    ) -> int:
        """
        Insert a single paper into Milvus
        
        Args:
            paper_id: Unique paper ID (INT64)
            title: Paper title
            abstract: Paper abstract
            categories: List of categories
            vector: Pre-computed embedding vector
        
        Returns:
            Number of inserted entities
        """
        # Prepare data
        categories_str = normalize_categories(categories)
        
        # Insert data
        result = self.collection.insert([
            [paper_id],
            [vector],
            [title],
            [abstract],
            [categories_str]
        ])
        
        # Flush to ensure data is written
        self.collection.flush()
        
        return len(result.primary_keys)
    
    def batch_insert(
        self,
        papers: List[Dict[str, Any]],
        batch_size: int = 2000
    ) -> int:
        """
        Insert multiple papers in batches
        
        Args:
            papers: List of paper dictionaries
            batch_size: Batch size for insertion
        
        Returns:
            Total number of inserted papers
        """
        total_inserted = 0
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            # Prepare batch data
            paper_ids = []
            vectors = []
            titles = []
            abstracts = []
            categories_list = []
            
            for paper in batch:
                paper_ids.append(paper["paper_id"])
                vectors.append(paper["vector"])
                titles.append(paper["title"])
                abstracts.append(paper["abstract"])
                categories_list.append(normalize_categories(paper.get("categories", [])))
            
            # Insert batch
            result = self.collection.insert([
                paper_ids,
                vectors,
                titles,
                abstracts,
                categories_list
            ])
            
            total_inserted += len(result.primary_keys)
            print(f"Inserted batch {i//batch_size + 1}: {len(result.primary_keys)} papers")
        
        # Flush final batch
        self.collection.flush()
        
        return total_inserted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "num_entities": self.collection.num_entities,
            "has_index": self.collection.has_index(),
            "collection_name": self.config.collection_name
        }
    
    def health_check(self) -> bool:
        """Check if Milvus is healthy and responsive"""
        try:
            # Check connection
            connections.get_connection_addr("default")
            
            # Check collection exists
            if not utility.has_collection(self.config.collection_name):
                return False
            
            # Get collection stats as simple health check
            _ = self.collection.num_entities
            
            return True
        except Exception:
            return False