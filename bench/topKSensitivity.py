import numpy as np
import time
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

    print("Running test search...")

    topKValues = [5, 10, 20, 40, 80, 160, 320]
    delays = []
    for topK in topKValues:
        search_params = {
            "metric_type": "IP",   # must match the index metric
            "params": {"ef": topK},
            }
        start = time.time()
        collection.search(
            data=[probe],
            anns_field="vector",
            param=search_params,
            limit=topK,
            output_fields=["paper_id", "title", "categories"],
        )
        end = time.time()
        delay = end - start
        delays.append(delay)

    print(f"topK={str(topKValues)}")
    print(f"delays={str(delays)}")


if __name__ == "__main__":
    main()
