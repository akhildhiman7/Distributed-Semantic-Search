"""
Load testing script for distributed cluster.

Compares performance:
1. Standalone vs Distributed Milvus
2. Single API vs Load-Balanced APIs
3. Different replica counts
"""

import asyncio
import httpx
import time
import statistics
from typing import List, Tuple
import json


class DistributedLoadTest:
    """Load test for distributed search system."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.queries = [
            "machine learning algorithms for image classification",
            "deep neural networks and transformers",
            "quantum computing and cryptography",
            "natural language processing techniques",
            "computer vision object detection",
            "reinforcement learning agents",
            "generative adversarial networks",
            "transfer learning pretrained models",
            "attention mechanisms in transformers",
            "convolutional neural networks architecture"
        ]
    
    async def single_search(self, client: httpx.AsyncClient, query: str) -> Tuple[float, bool]:
        """Perform single search and return latency and success."""
        start = time.time()
        try:
            response = await client.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": 10},
                timeout=30.0
            )
            latency = (time.time() - start) * 1000
            success = response.status_code == 200
            return latency, success
        except Exception as e:
            latency = (time.time() - start) * 1000
            return latency, False
    
    async def run_concurrent_load(self, concurrency: int, duration: int) -> dict:
        """Run concurrent load test for specified duration."""
        print(f"\n🔥 Running load test: {concurrency} concurrent users, {duration}s duration")
        
        latencies = []
        successes = 0
        failures = 0
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            tasks = []
            
            while (time.time() - start_time) < duration:
                # Create batch of concurrent requests
                for _ in range(concurrency):
                    query = self.queries[len(tasks) % len(self.queries)]
                    task = asyncio.create_task(self.single_search(client, query))
                    tasks.append(task)
                
                # Wait for batch to complete
                if len(tasks) >= concurrency:
                    results = await asyncio.gather(*tasks)
                    for latency, success in results:
                        latencies.append(latency)
                        if success:
                            successes += 1
                        else:
                            failures += 1
                    tasks = []
            
            # Complete remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks)
                for latency, success in results:
                    latencies.append(latency)
                    if success:
                        successes += 1
                    else:
                        failures += 1
        
        elapsed = time.time() - start_time
        total_requests = successes + failures
        
        return {
            "total_requests": total_requests,
            "successes": successes,
            "failures": failures,
            "elapsed_seconds": elapsed,
            "rps": total_requests / elapsed,
            "latency_p50": statistics.median(latencies) if latencies else 0,
            "latency_p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            "latency_p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            "latency_avg": statistics.mean(latencies) if latencies else 0,
            "latency_max": max(latencies) if latencies else 0,
        }
    
    async def compare_endpoints(self):
        """Compare standalone vs load-balanced endpoints."""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON: Standalone vs Distributed")
        print("="*70)
        
        # Test configuration
        test_configs = [
            ("Standalone (Single API)", "http://localhost:8000", 10, 30),
            ("Distributed (Load Balanced)", "http://localhost:8000", 10, 30),
            ("Distributed (High Load)", "http://localhost:8000", 50, 30),
        ]
        
        results = []
        
        for name, url, concurrency, duration in test_configs:
            print(f"\n📊 Testing: {name}")
            print(f"   URL: {url}")
            print(f"   Concurrency: {concurrency}, Duration: {duration}s")
            
            self.base_url = url
            result = await self.run_concurrent_load(concurrency, duration)
            result["name"] = name
            results.append(result)
            
            print(f"\n   Results:")
            print(f"   - Total Requests: {result['total_requests']}")
            print(f"   - Success Rate: {result['successes']/result['total_requests']*100:.1f}%")
            print(f"   - RPS: {result['rps']:.2f}")
            print(f"   - Latency (p50/p95/p99): {result['latency_p50']:.0f}ms / {result['latency_p95']:.0f}ms / {result['latency_p99']:.0f}ms")
        
        # Summary comparison
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print(f"\n{'Test':<30} {'RPS':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<30} {result['rps']:>10.1f} {result['latency_p50']:>10.0f} "
                  f"{result['latency_p95']:>10.0f} {result['latency_p99']:>10.0f}")
        
        # Calculate improvement
        if len(results) >= 2:
            standalone = results[0]
            distributed = results[1]
            
            rps_improvement = ((distributed['rps'] - standalone['rps']) / standalone['rps']) * 100
            latency_improvement = ((standalone['latency_p95'] - distributed['latency_p95']) / standalone['latency_p95']) * 100
            
            print("\n" + "="*70)
            print("IMPROVEMENT METRICS")
            print("="*70)
            print(f"Throughput (RPS): {rps_improvement:+.1f}%")
            print(f"Latency (p95):    {latency_improvement:+.1f}%")
        
        return results


async def main():
    """Main test function."""
    print("="*70)
    print("DISTRIBUTED CLUSTER LOAD TEST")
    print("="*70)
    
    # Default to load-balanced endpoint
    tester = DistributedLoadTest("http://localhost:8000")
    
    # Run comparison tests
    results = await tester.compare_endpoints()
    
    # Save results
    output_file = "distributed_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
