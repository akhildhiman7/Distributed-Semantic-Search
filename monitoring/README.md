# Monitoring & Benchmarking (Member 5)

This directory contains the Prometheus + Grafana stack plus a lightweight load test harness to validate the distributed semantic search system.

## Prometheus
- Config: `monitoring/prometheus/prometheus.yml`
- Alerts: `monitoring/prometheus/alerts.yml`
- Compose: `monitoring/prometheus/docker-compose.yml`

Start Prometheus only:
```bash
cd monitoring/prometheus
docker compose up -d
```
Scrapes:
- API: `localhost:8000` (and `api:8000` inside docker)
- Milvus: `localhost:9091` (and `milvus-standalone:9091` inside docker)

Alert rules cover: high error rate (>5%), high p95 latency (>1s), API down, Milvus down, low entity count.

## Grafana
- Compose: `monitoring/grafana/docker-compose.yml`
- Datasource provisioning: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- Dashboard provisioning: `monitoring/grafana/provisioning/dashboards/default.yml`
- Dashboards live in `monitoring/grafana/dashboards/` (API overview, search performance, Milvus metrics, system resources).

Start Grafana (bundled with a Prometheus container that reuses the config above):
```bash
cd monitoring/grafana
docker compose up -d
```
Login: `admin/admin` (change via env vars in compose).

Note: The Milvus and system resource dashboards expect Milvus metrics and node_exporter metrics respectively. Adjust PromQL if your Milvus metric names differ.

## Load/Benchmark Harness
- Script: `monitoring/loadtest/load_test.py`
- Requirements: `monitoring/loadtest/requirements.txt`

Example (60s, 20 concurrent users):
```bash
cd monitoring/loadtest
pip install -r requirements.txt
python load_test.py --host http://localhost:8000 --concurrency 20 --duration 60 --top-k 5
```
The script issues randomized search queries, gathers latency stats, and prints a brief summary (RPS, p50/p95/p99, error count).

## Quick Run Sequence
1) Start Milvus + API.
2) `docker compose up -d` in `monitoring/grafana` (brings up Prometheus + Grafana).
3) Visit Grafana at `http://localhost:3000`, select the provisioned dashboards.
4) Run the load test script to generate traffic and watch the dashboards/alerts.
