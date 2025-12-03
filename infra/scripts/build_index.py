from pymilvus import connections, Collection

from config import MilvusConfig


def main():
    cfg = MilvusConfig()

    connections.connect(
        alias="default",
        host=cfg.host,
        port=int(cfg.port),
        timeout=60,
    )
    collection = Collection(cfg.collection_name)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",  # use "L2" if your embeddings are not normalized
        "params": {"M": 16, "efConstruction": 200},
    }

    print("Creating index on field 'vector' with params:", index_params)
    collection.create_index(field_name="vector", index_params=index_params)

    print("Loading collection into memory...")
    collection.load()

    print("Index built and collection loaded.")


if __name__ == "__main__":
    main()
