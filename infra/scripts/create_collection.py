from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from config import MilvusConfig


def main():
    cfg = MilvusConfig()

    print(f"Connecting to Milvus at {cfg.host}:{cfg.port} ...")
    connections.connect(
        alias="default",
        host=cfg.host,
        port=int(cfg.port),
        timeout=60,
)

    if utility.has_collection(cfg.collection_name):
        print(f"Dropping existing collection: {cfg.collection_name}")
        utility.drop_collection(cfg.collection_name)

    fields = [
        FieldSchema(
            name="paper_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=cfg.dim,
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="abstract",
            dtype=DataType.VARCHAR,
            max_length=4096,
        ),
        FieldSchema(
            name="categories",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
    ]

    schema = CollectionSchema(fields, description="arXiv paper embeddings")

    print(f"Creating collection: {cfg.collection_name}")
    collection = Collection(name=cfg.collection_name, schema=schema)

    print("Collection created:")
    print(collection)


if __name__ == "__main__":
    main()
