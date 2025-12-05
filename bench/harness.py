from pymilvus import connections, Collection
import time

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Replace with your collection name
collection_name = "my_collection"
collection = Collection(name=collection_name)

# Example: Cold vs Warm latency
def measure_latency(query_vectors, top_k=10):
    latencies = []
    for vector in query_vectors:
        start = time.time()
        collection.search([vector], "embedding", top_k=top_k)
        latencies.append(time.time() - start)
    return latencies

if __name__ == "__main__":
    # Load your query vectors
    query_vectors = [[0.1, 0.2, 0.3]]  # Replace with real vectors
    latencies = measure_latency(query_vectors)
    print("Latencies:", latencies)
