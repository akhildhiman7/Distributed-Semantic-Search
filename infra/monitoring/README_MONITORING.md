# Monitoring (Prometheus + Grafana)

## Overview
This directory contains the monitoring setup for tracking FastAPI performance, Milvus metrics, and system resource usage for the Distributed Semantic Search project.

## How to run
From repo root (or `infra/`), start your main stack first:

```bash
cd infra
docker compose up -d
```

## The stack includes:

**Prometheus** – scrapes metrics

**Grafana** – visual dashboards

**Node Exporter** – collects system-level metrics

All services run inside the shared infra_default Docker network.

**🚀 How to Run**
Start the monitoring stack
```
cd monitoring
docker compose -f docker-compose-monitoring.yml up -d
```

**This starts:**

prometheus_monitoring
grafana_monitoring
node_exporter_monitoring

## Prometheus

Config file:
monitoring/prometheus.yml

Prometheus scrapes the following targets:

**Service	Target**
FastAPI	host.docker.internal:8000/metrics
Milvus	host.docker.internal:9091/metrics
MinIO	minio:9000/minio/v2/metrics/cluster
Node Exporter	node_exporter_monitoring:9100/metrics

Prometheus UI:
👉 http://localhost:9090

## Grafana

Provisioning directories:

**Datasources** → grafana/provisioning/datasources/

**Dashboards** → grafana/provisioning/dashboards/

**Dashboard JSON files** → grafana/dashboards/

Grafana UI:
👉 http://localhost:3000

Login: admin / admin

**Dashboards Included** - API Performance, Milvus Overview, System Metrics

🖥 Node Exporter

Node Exporter exposes system metrics:

http://localhost:9100/metrics