import numpy as np
import time
import random
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

    search_params = {
        "metric_type": "IP",   # must match the index metric
        "params": {"ef": 64},
    }

    print("Running test search...")
    interval = 1 / 100  # 100 RPS
    next_time = time.time()

    while True:
        probe = vectors[random.randrange(len(vectors))].tolist()
        collection.search(
            data=[probe],
            anns_field="vector",
            param=search_params,
            limit=5,
            output_fields=["paper_id", "title", "categories"],
        )
        next_time += interval
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
