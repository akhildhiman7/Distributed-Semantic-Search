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


## Benchmarks

### Cold vs Warm Latency Testing

Cold query: First query on the collection, data not loaded into memory (e.g. cache).
Warm query: Subsequent queries, collection already loaded in memory.

- Precondition 1: The system must be cleaned
  If not done so, please follow these steps:
```bash
cd infra
docker compose down
cd volumes
rm -rf etcd milvus minio
```
- Precondition 2: Prepare the data
```bash
cd infra
docker compose up -d
cd ..
python infra/scripts/create_collection.py
python infra/scripts/load_data.py
python infra/scripts/build_index.py
```
- Run Cold Warm Latency test
```bash
python bench/coldWarmLatency.py
```
- The outputs are 2 arrays (cold and warm) containing samples of latency for various searches.


### Top K Sensitivity Testing

Top_k determines how many nearest neighbors you want for a query.
Latency usually increases with top_k because more candidates need to be retrieved.

- Precondition 2 applies.
- Run top K test
```bash
python bench/topKSensitivity.py
```
- The test will run search for k values = [5, 10, 20, 40, 80, 160, 320]
- The outputs will be an array containing latency for each k values.

### Workload Testing

- Precondition 2 applies.
- Start workload test by running the following command
```bash
python bench/workloadTest.py
```
- Note that by default, it is running at 100 RPS.  It is possible to vary.


