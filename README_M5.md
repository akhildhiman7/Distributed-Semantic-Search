# Member 5

## Observability
### 1. Prometheus

- Configuration file: `infra/prometheus.yml`
- Docker Compose service: modified `docker-compose.yml`

### 2. Grafana dashboards
-  Dashboards: `infra/grafana/provisioning`

### 3. To run Prometheus and Grafana:

- Make sure Docker is running.
- Open a terminal and navigate to your project directory: cd project_path/infra
- Start the containers: docker compose up -d

### 4. Access the services:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000



##---------------Benchmarks--------------------------------------------------------------

- bench/locustfile.py, bench/harness.py: I'm not sure about this task, still working on it.

- pip install locust



