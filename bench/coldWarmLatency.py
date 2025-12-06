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

    search_params = {
        "metric_type": "IP",   # must match the index metric
        "params": {"ef": 64},
    }

    delaysCold = []
    delaysWarm = []
    for trial in range(0, 2):
        print("Start measuring cold delays...")
        for fileIndex, vf in enumerate(vec_files):
            first_vec_file = vec_files[fileIndex]
            print(f"Loading probe vector from {first_vec_file}")
            vectors = np.load(first_vec_file)
            probe = vectors[0].tolist()
            start = time.time()
            results = collection.search(
                data=[probe],
                anns_field="vector",
                param=search_params,
                limit=5,
                output_fields=["paper_id", "title", "categories"],
            )
            end = time.time()
            delays = end - start
            if trial == 0:
                delaysCold.append(delays)
            else:
                delaysWarm.append(delays)

            for hit in results[0]:
                print(
                    f"score={hit.score:.4f} "
                    f"paper_id={hit.entity.get('paper_id')} "
                    f"categories={hit.entity.get('categories')} "
                    f"title={hit.entity.get('title')[:80]!r}"
                )
    print(f"delaysCold= {str(delaysCold)}")
    print(f"delaysWarm= {str(delaysWarm)}")


if __name__ == "__main__":
    main()
