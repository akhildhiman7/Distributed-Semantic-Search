# Distributed Semantic Search – Member 3 Setup

This repo contains the infra and scripts to load arXiv paper embeddings into Milvus and run semantic search.

## 1. Prerequisites

- **Docker Desktop** installed and running
- **Python 3.10+**
- Install Python dependencies:

```bash
pip install -r infra/requirements.txt
```

## 2. Download embeddings & metadata

Member 2 provides the data via Google Drive.

1. From the repo root, create the folders:

```bash
mkdir -p embed/metadata embed/embeddings
```

2. Download the files from Google Drive and place them as:

- `embed/metadata/part-000*_metadata.parquet`
- `embed/embeddings/part-000*_embeddings.npy`

(Just keep the same filenames as in Drive.)

## 3. Start Milvus + etcd + MinIO (Docker)

From the `infra` folder:

```bash
cd infra
docker compose up -d
cd ..
```

You can check they’re running with:

```bash
cd infra
docker compose ps
cd ..
```

You should see services `etcd`, `milvus`, and `minio` all **Up**.

## 4. Initialize Milvus collection and load data

From the repo root:

```bash
# 1) Create (or recreate) the "papers" collection schema in Milvus
python infra/scripts/create_collection.py

# 2) Insert all embeddings + metadata into Milvus
python infra/scripts/load_data.py

# 3) Build a vector index and load the collection into memory
python infra/scripts/build_index.py
```

The `load_data.py` step may take a while the first time since it inserts ~500k rows.

## 5. Run a test search

```bash
python infra/scripts/smoke_search.py
```

You should see 5 search results printed with:

- similarity `score`
- `paper_id`
- `categories`
- the paper `title`

This confirms Milvus is set up correctly and the data + index are usable.

## 6. Next time you restart (day-to-day usage)

You **do not** need to reload all the data every time.

For a normal session:

```bash
cd infra
docker compose up -d
cd ..
python infra/scripts/smoke_search.py
```

`smoke_search.py` will:

- connect to Milvus
- load the existing `papers` collection into memory
- run a small semantic search query as a sanity check.
