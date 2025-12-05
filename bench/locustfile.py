from locust import User, task, between
import random
import json
from pymilvus import connections, Collection

# Connect to Milvus directly
connections.connect(host="localhost", port="19530")
collection = Collection("my_collection")

# Load canned queries
with open("queries.json", "r") as f:
    QUERIES = json.load(f)

class MilvusUser(User):
    wait_time = between(1, 3)  # seconds

    @task
    def query_milvus(self):
        query = random.choice(QUERIES)
        # Perform search directly via pymilvus
        collection.search([query["vector"]], "embedding", top_k=query.get("top_k", 10))

