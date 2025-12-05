# Member 5

##---------------Observability-----------------------------------------------------------
1. Observability

1.1. Prometheus: infra/prometheus.yml, modified docker-compose.yml
1.2. Grafana dashboards: infra/grafana/provisioning

2. To run Prometheus and Grafana:

2.1. Make sure Docker is running.
2.2. Open a terminal and navigate to your project directory: cd project_path/infra
2.3. Start the containers: docker compose up -d

2. Access the services:

Prometheus: http://localhost:9090
Grafana: http://localhost:3000

##---------------Benchmarks--------------------------------------------------------------

bench/locustfile.py, bench/harness.py: I'm not sure about this task, still working on it.

pip install locust



