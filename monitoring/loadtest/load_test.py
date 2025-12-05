"""Lightweight async load generator for the semantic search API."""
import argparse
import asyncio
import random
import time
from typing import List

import httpx


QUERIES = [
    "neural networks for image classification",
    "quantum computing algorithms",
    "natural language processing transformers",
    "graph neural networks for recommendation",
    "reinforcement learning for robotics",
    "computer vision object detection",
    "large language models for code",
    "time series forecasting",
    "anomaly detection in logs",
    "diffusion models for images",
]


class Stats:
    def __init__(self) -> None:
        self.latencies: List[float] = []
        self.errors: int = 0
        self.requests: int = 0
        self._lock = asyncio.Lock()

    async def record(self, latency: float, error: bool = False) -> None:
        async with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            else:
                self.latencies.append(latency)


async def worker(client: httpx.AsyncClient, stats: Stats, end_time: float, top_k: int) -> None:
    """Send requests until end_time is reached."""
    while time.time() < end_time:
        payload = {
            "query": random.choice(QUERIES),
            "top_k": top_k,
        }
        start = time.perf_counter()
        try:
            resp = await client.post("/search", json=payload, timeout=10.0)
            latency = time.perf_counter() - start
            error = resp.status_code >= 400
            await stats.record(latency, error=error)
        except Exception:
            latency = time.perf_counter() - start
            await stats.record(latency, error=True)


def percentile(values: List[float], p: float) -> float:
    """Compute percentile for a list of floats."""
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def run(host: str, concurrency: int, duration: int, top_k: int) -> None:
    stats = Stats()
    end_time = time.time() + duration
    async with httpx.AsyncClient(base_url=host) as client:
        tasks = [
            asyncio.create_task(worker(client, stats, end_time, top_k))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)

    success = len(stats.latencies)
    total = stats.requests
    rps = total / duration if duration else 0
    p50 = percentile(stats.latencies, 50)
    p95 = percentile(stats.latencies, 95)
    p99 = percentile(stats.latencies, 99)

    print("Load test complete")
    print(f"Host: {host}")
    print(f"Duration: {duration}s  Concurrency: {concurrency}  top_k: {top_k}")
    print(f"Requests: {total}  Success: {success}  Errors: {stats.errors}  RPS: {rps:.2f}")
    print(f"Latency (s): p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple load generator for the semantic search API.")
    parser.add_argument("--host", default="http://localhost:8000", help="API host (default: http://localhost:8000)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="top_k parameter for search")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.host, args.concurrency, args.duration, args.top_k))
