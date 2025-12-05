import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from api.main import app


class TestMilvusAdapter:
    """Mock Milvus adapter for testing"""
    
    def __init__(self):
        self.collection = Mock()
        self.config = Mock(collection_name="papers", dim=384)
    
    def search(self, vector, top_k=10, categories=None, expr=None):
        # Return mock search results
        return [
            {
                "paper_id": 123456,
                "title": "Test Paper on Machine Learning",
                "abstract": "This is a test abstract about machine learning.",
                "categories": ["AI", "ML"],
                "score": 0.95
            },
            {
                "paper_id": 789012,
                "title": "Another Test Paper",
                "abstract": "Another test abstract.",
                "categories": ["Computer Science"],
                "score": 0.85
            }
        ]
    
    def insert_paper(self, paper_id, title, abstract, categories, vector):
        return 1
    
    def get_stats(self):
        return {
            "num_entities": 510203,
            "has_index": True,
            "collection_name": "papers"
        }
    
    def health_check(self):
        return True


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_adapter():
    """Create mock Milvus adapter"""
    return TestMilvusAdapter()


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Paper Search API"


def test_health_endpoint(client, mock_adapter):
    """Test health endpoint"""
    # Patch the milvus_adapter in the app
    from api import main
    main.milvus_adapter = mock_adapter
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["milvus_connected"] == True


def test_stats_endpoint(client, mock_adapter):
    """Test stats endpoint"""
    from api import main
    main.milvus_adapter = mock_adapter
    
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "num_papers" in data
    assert data["num_papers"] == 510203


def test_search_endpoint(client, mock_adapter):
    """Test search endpoint"""
    from api import main
    main.milvus_adapter = mock_adapter
    main.embedder = Mock()
    main.embedder.encode.return_value = [0.1] * 384  # Mock embedding
    
    search_data = {
        "query": "machine learning",
        "top_k": 2
    }
    
    response = client.post("/search", json=search_data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["paper_id"] == 123456
    assert "machine learning" in data[0]["title"].lower()


def test_search_with_categories(client, mock_adapter):
    """Test search with category filter"""
    from api import main
    main.milvus_adapter = mock_adapter
    main.embedder = Mock()
    main.embedder.encode.return_value = [0.1] * 384
    
    search_data = {
        "query": "machine learning",
        "top_k": 2,
        "categories": ["AI", "ML"]
    }
    
    response = client.post("/search", json=search_data)
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0


def test_insert_endpoint(client, mock_adapter):
    """Test insert endpoint"""
    from api import main
    main.milvus_adapter = mock_adapter
    
    paper_data = {
        "paper_id": 999999,
        "title": "Test Paper",
        "abstract": "Test abstract for unit testing.",
        "categories": ["Test", "Unit"],
        "vector": [0.1] * 384
    }
    
    response = client.post("/insert", json=paper_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["paper_id"] == 999999


def test_insert_invalid_vector(client, mock_adapter):
    """Test insert with invalid vector dimension"""
    from api import main
    main.milvus_adapter = mock_adapter
    
    paper_data = {
        "paper_id": 999999,
        "title": "Test Paper",
        "abstract": "Test abstract",
        "categories": ["Test"],
        "vector": [0.1] * 100  # Wrong dimension
    }
    
    response = client.post("/insert", json=paper_data)
    assert response.status_code == 400
    data = response.json()
    assert "dimension mismatch" in data["detail"].lower()


def test_rate_limiting(client):
    """Test rate limiting middleware"""
    # Make many rapid requests
    for i in range(105):  # More than 100 requests (the default limit)
        response = client.get("/health")
    
    # Should get 429 Too Many Requests
    assert response.status_code == 429
    data = response.json()
    assert "Too Many Requests" in data["error"]
    assert "X-RateLimit-Limit" in response.headers

def test_batch_insert(client, mock_adapter):
    """Test batch insert endpoint"""
    from api import main
    main.milvus_adapter = mock_adapter
    
    papers_data = [
        {
            "paper_id": 100001,
            "title": "Paper 1",
            "abstract": "Abstract 1",
            "categories": ["AI"],
            "vector": [0.1] * 384
        },
        {
            "paper_id": 100002,
            "title": "Paper 2",
            "abstract": "Abstract 2",
            "categories": ["ML"],
            "vector": [0.2] * 384
        }
    ]
    
    response = client.post("/batch_insert", json=papers_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_inserted"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])