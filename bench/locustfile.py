from locust import User, task, between, constant_throughput
from pymilvus import connections, Collection
from config import MilvusConfig, EmbeddingPaths
import numpy as np
import random


class MilvusUser(User):
    wait_time = None

    def __init__(self, environment):
        super().__init__(environment)
        self.cfg = MilvusConfig()
        connections.connect(host="localhost", port="19530")
        self.collection = Collection(self.cfg.collection_name)
        print("Loading collection into memory (if not already loaded)...")
        self.collection.load()
        self.paths = EmbeddingPaths()
        vec_files = self.paths.vector_paths()
        if not vec_files:
            raise SystemExit("No vector files found; check EMBED_ROOT and *_GLOB in .env")
        self.vectors = np.load(vec_files[0])

    @task
    def query_milvus(self):

        probe = self.vectors[random.randrange(len(self.vectors))].tolist()

        search_params = {
            "metric_type": "IP",   # must match the index metric
            "params": {"ef": 64},
        }
        try:
            self.collection.search(
                data=[probe],
                anns_field="vector",
                param=search_params,
                limit=100,
                output_fields=["paper_id", "title", "categories"],
                timeout=10,
            )
        except Exception as e:
            self.environment.events.request.fire(
                request_type="milvus",
                name="search",
                response_time=0,
                response_length=0,
                exception=e,
            )

