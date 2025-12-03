import numpy as np

from pymilvus import connections, Collection

from config import MilvusConfig, EmbeddingPaths


def main():
    cfg = MilvusConfig()
    paths = EmbeddingPaths()

    connections.connect(
        alias="default",
        host=cfg.host,
        port=int(cfg.port),
        timeout=60,
    )
    collection = Collection(cfg.collection_name)

    # Ensure collection is loaded into memory (needed after Milvus restart)
    print("Loading collection into memory (if not already loaded)...")
    collection.load()

    vec_files = paths.vector_paths

    if not vec_files:
        raise SystemExit("No vector files found; check EMBED_ROOT and *_GLOB in .env")

    first_vec_file = vec_files[0]
    print(f"Loading probe vector from {first_vec_file}")
    vectors = np.load(first_vec_file)
    probe = vectors[0].tolist()

    search_params = {
        "metric_type": "IP",   # must match the index metric
        "params": {"ef": 64},
    }

    print("Running test search...")
    results = collection.search(
        data=[probe],
        anns_field="vector",
        param=search_params,
        limit=5,
        output_fields=["paper_id", "title", "categories"],
    )

    for hit in results[0]:
        print(
            f"score={hit.score:.4f} "
            f"paper_id={hit.entity.get('paper_id')} "
            f"categories={hit.entity.get('categories')} "
            f"title={hit.entity.get('title')[:80]!r}"
        )


if __name__ == "__main__":
    main()
