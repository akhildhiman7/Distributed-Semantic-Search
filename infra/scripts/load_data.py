from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymilvus import connections, Collection

from config import MilvusConfig, EmbeddingPaths


BATCH_SIZE = 2000  # tune if needed


def normalize_categories(values):
    """Ensure categories are strings like 'cs.LG,cs.AI'."""
    norm = []
    for v in values:
        if isinstance(v, (list, tuple, set)):
            norm.append(",".join(sorted(map(str, v))))
        else:
            norm.append("" if v is None else str(v))
    return norm


def main():
    milvus_cfg = MilvusConfig()
    embed_paths = EmbeddingPaths()

    connections.connect(
        alias="default",
        host=milvus_cfg.host,
        port=int(milvus_cfg.port),
        timeout=60,
    )

    collection = Collection(milvus_cfg.collection_name)

    meta_files = embed_paths.metadata_paths
    vec_files = embed_paths.vector_paths

    if len(meta_files) == 0:
        raise SystemExit(f"No metadata files found under {embed_paths.root}")
    if len(meta_files) != len(vec_files):
        raise SystemExit(
            f"Mismatch: {len(meta_files)} metadata vs {len(vec_files)} vector files"
        )

    print("Loading data into Milvus...")
    total_inserted = 0

    for meta_path, vec_path in tqdm(
        list(zip(meta_files, vec_files)), desc="Partitions"
    ):
        print(f"\nPartition meta={meta_path.name} vectors={vec_path.name}")

        df = pd.read_parquet(meta_path)
        vectors = np.load(vec_path)

        if len(df) != len(vectors):
            raise ValueError(
                f"Row mismatch in {meta_path} and {vec_path}: "
                f"{len(df)} vs {len(vectors)}"
            )

        # Make sure IDs are int64
        paper_ids = df["paper_id"].astype("int64").tolist()
        titles = df["title"].astype(str).tolist()
        abstracts = df["abstract"].astype(str).tolist()
        categories = normalize_categories(df["categories"].tolist())

        n = len(df)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            batch = {
                "paper_id": paper_ids[start:end],
                "vector": vectors[start:end].tolist(),
                "title": titles[start:end],
                "abstract": abstracts[start:end],
                "categories": categories[start:end],
            }

            # Column-based insert: list of field-value lists in schema order
            collection.insert([
                batch["paper_id"],
                batch["vector"],
                batch["title"],
                batch["abstract"],
                batch["categories"],
            ])

            total_inserted += (end - start)


    print("Flushing collection...")
    collection.flush()
    print(f"Total entities inserted: {total_inserted}")
    print(f"Milvus reports {collection.num_entities} entities.")


if __name__ == "__main__":
    main()
