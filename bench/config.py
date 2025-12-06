from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)


@dataclass
class MilvusConfig:
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: str = os.getenv("MILVUS_PORT", "19530")
    collection_name: str = os.getenv("MILVUS_COLLECTION", "papers")
    dim: int = int(os.getenv("VECTOR_DIM", "384"))


@dataclass
class EmbeddingPaths:
    root: Path = Path(os.getenv("EMBED_ROOT", "embed"))
    metadata_glob: str = os.getenv("EMBED_METADATA_GLOB", "metadata/part-*.parquet")
    vectors_glob: str = os.getenv("EMBED_VECTORS_GLOB", "embeddings/part-*.npy")

    @property
    def metadata_paths(self):
        return sorted(self.root.glob(self.metadata_glob))

    @property
    def vector_paths(self):
        return sorted(self.root.glob(self.vectors_glob))
