"""
Test script for FastAPI service.
Tests all endpoints and validates responses.
"""
import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def test_root():
    """Test root endpoint."""
    print("\n" + "="*60)
    print("TEST: Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Message: {data['message']}")
    print(f"✓ Version: {data['version']}")
    print(f"✓ Docs available at: {data['docs']}")


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Status: {data['status']}")
    print(f"✓ Milvus Connected: {data['milvus_connected']}")
    print(f"✓ Collection Loaded: {data['collection_loaded']}")
    print(f"✓ Total Entities: {data['total_entities']:,}")
    print(f"✓ Model Loaded: {data['model_loaded']}")
    print(f"✓ API Version: {data['api_version']}")
    
    assert data['status'] == 'healthy'
    assert data['milvus_connected'] is True
    assert data['collection_loaded'] is True
    assert data['total_entities'] > 500000


def test_stats():
    """Test stats endpoint."""
    print("\n" + "="*60)
    print("TEST: Collection Statistics")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/stats")
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Collection: {data['collection_name']}")
    print(f"✓ Total Entities: {data['total_entities']:,}")
    print(f"✓ Index Type: {data['index_type']}")
    print(f"✓ Vector Dimension: {data['vector_dim']}")
    print(f"✓ Metric Type: {data['metric_type']}")
    print(f"✓ Model: {data['model_name']}")


def test_search_basic():
    """Test basic search."""
    print("\n" + "="*60)
    print("TEST: Basic Search")
    print("="*60)
    
    query = "neural networks deep learning"
    payload = {
        "query": query,
        "top_k": 3
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    end_time = time.time()
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Query: {data['query']}")
    print(f"✓ Results Found: {data['total_results']}")
    print(f"✓ Search Latency: {data['latency_ms']:.2f}ms")
    print(f"✓ Total API Latency: {(end_time - start_time) * 1000:.2f}ms")
    print(f"\nTop Result:")
    print(f"  - Title: {data['results'][0]['title']}")
    print(f"  - Score: {data['results'][0]['score']:.4f}")
    print(f"  - Categories: {data['results'][0]['categories']}")
    
    assert len(data['results']) == 3
    assert all(r['score'] > 0.5 for r in data['results'])


def test_search_with_filters():
    """Test search with score threshold."""
    print("\n" + "="*60)
    print("TEST: Search with Score Filter")
    print("="*60)
    
    query = "quantum computing algorithms"
    payload = {
        "query": query,
        "top_k": 5,
        "min_score": 0.7
    }
    
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Query: {data['query']}")
    print(f"✓ Min Score Threshold: {payload['min_score']}")
    print(f"✓ Results Found: {data['total_results']}")
    print(f"✓ Search Latency: {data['latency_ms']:.2f}ms")
    
    if data['results']:
        print(f"\nTop Result:")
        print(f"  - Title: {data['results'][0]['title']}")
        print(f"  - Score: {data['results'][0]['score']:.4f}")
        
        # Verify all results meet threshold
        assert all(r['score'] >= payload['min_score'] for r in data['results'])
        print(f"✓ All results have score >= {payload['min_score']}")
    else:
        print("⚠ No results met the score threshold")


def test_search_with_category():
    """Test search with category filter."""
    print("\n" + "="*60)
    print("TEST: Search with Category Filter")
    print("="*60)
    
    query = "transformers for NLP"
    payload = {
        "query": query,
        "top_k": 3,
        "categories": ["cs.CL", "cs.LG"]
    }
    
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Query: {data['query']}")
    print(f"✓ Category Filter: {payload['categories']}")
    print(f"✓ Results Found: {data['total_results']}")
    print(f"✓ Search Latency: {data['latency_ms']:.2f}ms")
    
    if data['results']:
        print(f"\nTop Result:")
        print(f"  - Title: {data['results'][0]['title']}")
        print(f"  - Score: {data['results'][0]['score']:.4f}")
        print(f"  - Categories: {data['results'][0]['categories']}")


def test_validation_error():
    """Test validation error handling."""
    print("\n" + "="*60)
    print("TEST: Input Validation")
    print("="*60)
    
    # Test short query
    payload = {"query": "ai", "top_k": 5}
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    
    print(f"✓ Short query validation: {response.status_code}")
    assert response.status_code == 422  # Validation error
    
    # Test invalid top_k
    payload = {"query": "machine learning", "top_k": 200}
    response = requests.post(f"{API_BASE_URL}/search", json=payload)
    
    print(f"✓ Invalid top_k validation: {response.status_code}")
    assert response.status_code == 422


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("FASTAPI SERVICE VALIDATION TESTS")
    print("="*60)
    print(f"API Base URL: {API_BASE_URL}")
    
    try:
        # Test connectivity
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        print(f"✓ API is accessible")
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: Cannot connect to API at {API_BASE_URL}")
        print("Make sure the API service is running:")
        print("  cd api && python main.py")
        return
    
    # Run tests
    tests = [
        test_root,
        test_health,
        test_stats,
        test_search_basic,
        test_search_with_filters,
        test_search_with_category,
        test_validation_error
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ TEST FAILED: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(tests)}")
    else:
        print("✓ ALL TESTS PASSED!")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
