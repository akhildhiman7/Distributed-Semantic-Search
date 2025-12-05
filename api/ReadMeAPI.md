 Architecture

text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI API   │────▶│   Milvus DB     │
│   (localhost:8501)│     │   (localhost:8000)│     │   (Vector Store)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                         │
                              │                         │
                        ┌─────▼─────┐             ┌─────▼─────┐
                        │  Rate     │             │   etcd    │
                        │  Limiting │             │  (Meta)   │
                        └───────────┘             └───────────┘
                              │                         │
                        ┌─────▼─────┐             ┌─────▼─────┐
                        │   CORS    │             │   minio   │
                        │  (Web)    │             │ (Storage) │
                        └───────────┘             └───────────┘
 Installation

1. Prerequisites

Python 3.9+
Docker & Docker Compose
Git
2. Clone and Setup

bash
git clone <repository-url>
cd Distributed-Semantic-Search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Environment Configuration

Create .env file:

env
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=papers
VECTOR_DIM=384

# Embedding Paths
EMBED_ROOT=embed
EMBED_METADATA_GLOB=metadata/part-*.parquet
EMBED_VECTORS_GLOB=embeddings/part-*.npy

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2


-------------------------------------------------------------------------------

Quick Start

1. Start Database

bash
cd infra
docker-compose up -d
cd ..
2. Start FastAPI Backend

bash
source venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
3. Start Streamlit UI (Optional)

bash
streamlit run ui/app.py
🔧 API Endpoints

Core Endpoints

GET / - API information
GET /health - Health check with rate limit headers
GET /stats - Collection statistics
POST /search - Semantic paper search
POST /insert - Insert new paper
POST /batch_insert - Insert multiple papers
Search Example

bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning neural networks",
    "top_k": 5,
    "categories": ["cs.AI", "stat.ML"],
    "min_score": 0.3
  }'
Insert Example

bash
curl -X POST "http://localhost:8000/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": 1000001,
    "title": "Test Paper",
    "abstract": "This is a test abstract...",
    "categories": ["cs.AI", "cs.LG"],
    "vector": [0.1, 0.2, ...] # 384-dimensional vector
  }'
Rate Limiting

The API implements rate limiting (100 requests/minute) with headers:

X-RateLimit-Limit: Maximum requests per minute
X-RateLimit-Remaining: Remaining requests
X-RateLimit-Reset: Reset timestamp (Unix)
Testing

Run unit tests:

bash
pytest tests/test_api.py -v
Test coverage includes:

API endpoints
Rate limiting
Error handling
Mocked Milvus adapter
Project Structure

text
Distributed-Semantic-Search/
├── api/                    # FastAPI backend
│   ├── main.py           # FastAPI application
│   ├── milvus_adapter.py # Milvus operations
│   ├── models.py         # Pydantic schemas
│   ├── middleware.py     # Rate limiting
│   └── dependencies.py   # Utilities
├── ui/                    # Streamlit frontend
│   └── app.py           # Web interface
├── tests/                # Unit tests
│   └── test_api.py      # Test suite
├── infra/                # Infrastructure
│   └── docker-compose.yml # Milvus stack
├── embed/                # Pre-computed embeddings
├── config.py            # Configuration
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables


