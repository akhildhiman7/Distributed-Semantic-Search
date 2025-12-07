# Monitoring (Prometheus + Grafana)

## Overview
This folder contains a lightweight monitoring stack that connects to your existing `infra` Docker network so Prometheus can scrape Milvus and other services.

## How to run
From repo root (or `infra/`), start your main stack first:

```bash
cd infra
docker compose up -d