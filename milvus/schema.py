"""
Milvus Collection Schema Definition
Defines the schema for the arxiv_papers collection with vector embeddings and metadata
"""

from pymilvus import CollectionSchema, FieldSchema, DataType
from config import COLLECTION_NAME, VECTOR_DIM, METRIC_TYPE, NUM_SHARDS

def create_collection_schema():
    """
    Create the collection schema for arxiv papers
    
    Schema includes:
    - paper_id: Unique identifier (primary key)
    - embedding: 384-dimensional vector
    - title: Paper title
    - abstract: Paper abstract
    - categories: ArXiv categories
    - text_length: Length of searchable text
    - has_full_data: Whether paper has complete data
    """
    
    fields = [
        # Primary key field
        FieldSchema(
            name="paper_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=False,
            description="Unique paper identifier (e.g., arxiv ID)"
        ),
        
        # Vector embedding field
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
            description="384-dimensional sentence embedding"
        ),
        
        # Metadata fields
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Paper title"
        ),
        
        FieldSchema(
            name="abstract",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Paper abstract"
        ),
        
        FieldSchema(
            name="categories",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="ArXiv categories (space-separated)"
        ),
        
        FieldSchema(
            name="text_length",
            dtype=DataType.INT32,
            description="Length of searchable text"
        ),
        
        FieldSchema(
            name="has_full_data",
            dtype=DataType.BOOL,
            description="Whether paper has complete title and abstract"
        )
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description=f"ArXiv papers with semantic embeddings",
        enable_dynamic_field=False  # Strict schema enforcement
    )
    
    return schema


def get_collection_properties():
    """
    Get collection creation properties for distributed setup
    """
    return {
        "collection.ttl.seconds": 0,  # 0 = no auto-deletion
        "shards_num": NUM_SHARDS,  # Number of shards for horizontal scaling
    }


def print_schema_info():
    """
    Print schema information for validation
    """
    schema = create_collection_schema()
    
    print(f"\n{'='*60}")
    print(f"Collection Schema: {COLLECTION_NAME}")
    print(f"{'='*60}")
    print(f"Vector Dimension: {VECTOR_DIM}")
    print(f"Metric Type: {METRIC_TYPE}")
    print(f"Number of Shards: {NUM_SHARDS}")
    print(f"\nFields:")
    
    for field in schema.fields:
        field_type = str(field.dtype).replace("DataType.", "")
        print(f"  - {field.name:20} {field_type:15} {field.description}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_schema_info()
