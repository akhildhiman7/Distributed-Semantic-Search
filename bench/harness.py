import time
import csv
import random
import os
import json
from pymilvus import connections, Collection

# ==========================
# Configuration
# ==========================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

COLLECTION_NAME = "my_collection"  # change to your collection name
REPORTS_DIR = "bench/reports/harness/"

TOP_K_VALUES = [5, 10, 20]
INDEX_TYPES = ["HNSW", "IVF_FLAT"]  # adjust as needed

# ==========================
# Load canned queries
# ==========================
with open("bench/queries.json", "r") as f:
    QUERIES = json.load(f)

# Ensure report folder exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==========================
# Connect to Milvus
# ==========================
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)

# ==========================
# Helper functions
# ==========================
def measure_latency(vector, top_k):
    """Measure query latency in seconds"""
    start = time.time()
    _ = collection.search(
        data=[vector],
        anns_field="embedding",        # name of your vector field
        param={"metric_type": "L2"},   # search parameters
        limit=top_k                    # top_k
    )
    return time.time() - start

def benchmark_queries():
    results = []

    for index_type in INDEX_TYPES:
        print(f"Testing index: {index_type}")
        # TODO: Apply index type to collection if needed

        for query_id, query in enumerate(QUERIES):
            vector = query["vector"]

            # Cold latency (first query)
            cold_latency = measure_latency(vector, TOP_K_VALUES[0])

            # Warm latency (second query)
            warm_latency = measure_latency(vector, TOP_K_VALUES[0])

            for top_k in TOP_K_VALUES:
                latency = measure_latency(vector, top_k)
                results.append({
                    "query_id": query_id,
                    "index_type": index_type,
                    "top_k": top_k,
                    "cold_latency": cold_latency,
                    "warm_latency": warm_latency,
                    "query_vector": vector
                })

            if (query_id + 1) % 10 == 0:
                print(f"Processed {query_id + 1}/{len(QUERIES)} queries")

    return results

def save_results(results):
    keys = ["query_id", "index_type", "top_k", "cold_latency", "warm_latency", "query_vector"]
    report_file = os.path.join(REPORTS_DIR, "benchmark_results.csv")
    with open(report_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {report_file}")

# ==========================
# Run benchmark
# ==========================
if __name__ == "__main__":
    results = benchmark_queries()
    save_results(results)
