# CSearch NLP — Detailed Implementation Steps

This document houses the raw deployment configurations, K8s manifests, infrastructure templates, and Phase-by-Phase orchestration steps for the NLP Vector Database Pipeline. For a conceptual overview, please refer to `README.md`.

## 1. Qdrant Deployment on Mars

### Namespace

```yaml
# k8s/csearch-nlp/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: csearch-nlp
```

### Persistent Volume (hostPath-backed SSD for Dev)

Because the development server (`mars`) uses local SSDs, we use `hostPath` to eliminate NFS overhead. Production deployment (`netcup`) storage will be evaluated separately.

```yaml
# k8s/csearch-nlp/qdrant-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qdrant-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  hostPath:
    path: /mnt/data/qdrant
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
  namespace: csearch-nlp
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

### Full-text cache PV

```yaml
# k8s/csearch-nlp/fulltext-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nlp-fulltext-pv
spec:
  capacity:
    storage: 20Gi
  accessModes:
    - ReadWriteMany
  storageClassName: standard
  hostPath:
    path: /mnt/data/nlp-fulltext
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nlp-fulltext-pvc
  namespace: csearch-nlp
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
```

### Qdrant StatefulSet

```yaml
# k8s/csearch-nlp/qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: csearch-nlp
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      nodeSelector:
        node: worker1
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: qdrant
          image: qdrant/qdrant:v1.12.1
          ports:
            - name: rest
              containerPort: 6333
            - name: grpc
              containerPort: 6334
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
          resources:
            requests:
              memory: "1.5Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1"
          env:
            - name: QDRANT__SERVICE__GRPC_PORT
              value: "6334"
          livenessProbe:
            httpGet:
              path: /healthz
              port: 6333
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /readyz
              port: 6333
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: qdrant-storage
          persistentVolumeClaim:
            claimName: qdrant-pvc
```

### Qdrant Service

```yaml
# k8s/csearch-nlp/qdrant-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: csearch-nlp
spec:
  selector:
    app: qdrant
  ports:
    - name: rest
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
  type: ClusterIP
```

---

## 2. NLP API Deployment on Mars

### Secrets

```yaml
# k8s/csearch-nlp/secrets.yaml (template — apply manually, do not commit real values)
apiVersion: v1
kind: Secret
metadata:
  name: csearch-nlp-secrets
  namespace: csearch-nlp
type: Opaque
stringData:
  pg-connection-string: "postgresql://postgres:postgres@postgres-service.default.svc.cluster.local:5432/csearch"
  embedding-api-key: "ollama-local"
  llm-api-key: "ollama-local"
```

### ConfigMap

```yaml
# k8s/csearch-nlp/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: csearch-nlp-config
  namespace: csearch-nlp
data:
  QDRANT_HOST: "qdrant.csearch-nlp.svc.cluster.local"
  QDRANT_PORT: "6333"
  QDRANT_COLLECTION: "bill_chunks"
  EMBEDDING_MODEL: "nomic-embed-text"
  LLM_MODEL: "qwen2.5:7b"
  RAG_VECTOR_TOP_K: "40"
  RAG_RERANK_TOP_K: "10"
  RAG_SCORE_THRESHOLD: "0.35"
  RRF_K: "60"
  REDIS_URL: "redis://redis-service.default.svc.cluster.local:6379/1"
  CACHE_EMBEDDING_TTL: "86400"
  CACHE_SEARCH_TTL: "3600"
  CACHE_LLM_TTL: "3600"
  FULLTEXT_CACHE_DIR: "/data/fulltext"
  GOVINFO_RATE_LIMIT: "10"
```

### API Deployment

```yaml
# k8s/csearch-nlp/nlp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csearch-nlp-api
  namespace: csearch-nlp
spec:
  replicas: 1                    # single replica on 4CPU/8GB VPS
  selector:
    matchLabels:
      app: csearch-nlp-api
  template:
    metadata:
      labels:
        app: csearch-nlp-api
    spec:
      nodeSelector:
        node: worker1
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
      containers:
        - name: api
          image: 10.0.0.3:30252/csearch-nlp:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: csearch-nlp-config
          env:
            - name: PG_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: csearch-nlp-secrets
                  key: pg-connection-string
            - name: EMBEDDING_API_KEY
              valueFrom:
                secretKeyRef:
                  name: csearch-nlp-secrets
                  key: embedding-api-key
            - name: LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: csearch-nlp-secrets
                  key: llm-api-key
          volumeMounts:
            - name: fulltext-cache
              mountPath: /data/fulltext
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
      volumes:
        - name: fulltext-cache
          persistentVolumeClaim:
            claimName: nlp-fulltext-pvc
```

### API Service

```yaml
# k8s/csearch-nlp/nlp-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: csearch-nlp-api
  namespace: csearch-nlp
spec:
  selector:
    app: csearch-nlp-api
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

### Nightly Sync CronJob

```yaml
# k8s/csearch-nlp/nlp-sync-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: csearch-nlp-sync
  namespace: csearch-nlp
spec:
  schedule: "30 0 * * *"          # 00:30 daily, after goscraper finishes at midnight
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          nodeSelector:
            node: worker1
          securityContext:
            runAsUser: 1000
            runAsGroup: 1000
          containers:
            - name: nlp-sync
              image: 10.0.0.3:30252/csearch-nlp:latest
              imagePullPolicy: Always
              command: ["python", "-m", "csearch_nlp.pipeline", "sync"]
              envFrom:
                - configMapRef:
                    name: csearch-nlp-config
              env:
                - name: PG_CONNECTION_STRING
                  valueFrom:
                    secretKeyRef:
                      name: csearch-nlp-secrets
                      key: pg-connection-string
                - name: EMBEDDING_API_KEY
                  valueFrom:
                    secretKeyRef:
                      name: csearch-nlp-secrets
                      key: embedding-api-key
              volumeMounts:
                - name: fulltext-cache
                  mountPath: /data/fulltext
              resources:
                requests:
                  memory: "512Mi"
                  cpu: "250m"
                limits:
                  memory: "1Gi"
                  cpu: "500m"
          volumes:
            - name: fulltext-cache
              persistentVolumeClaim:
                claimName: nlp-fulltext-pvc
          restartPolicy: OnFailure
```

---

## 3. Project File Structure

```text
csearch-nlp/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml              # Local dev: Qdrant + Redis + API
│
├── csearch_nlp/
│   ├── __init__.py
│   ├── config.py                   # Env vars, model names, thresholds
│   │
│   ├── api/
│   │   ├── server.py               # FastAPI app
│   │   ├── routes.py               # POST /api/nlp/search, GET /health
│   │   └── models.py               # Pydantic request/response schemas
│   │
│   ├── rag/
│   │   ├── orchestrator.py         # Main pipeline
│   │   ├── query_classifier.py     # Filter extraction, intent classification
│   │   ├── embedder.py             # Embedding API wrapper
│   │   ├── retriever.py            # Qdrant + PG keyword search
│   │   ├── reranker.py             # Cross-encoder
│   │   ├── fusion.py               # Reciprocal Rank Fusion
│   │   ├── hydrator.py             # PG bill metadata fetch
│   │   ├── prompt_builder.py       # LLM prompt construction
│   │   └── generator.py            # Claude streaming client
│   │
│   ├── pipeline/
│   │   ├── __main__.py             # CLI: batch / sync / reembed
│   │   ├── fetcher.py              # GovInfo XML downloader
│   │   ├── chunker.py              # XML-aware section splitter
│   │   ├── batcher.py              # Embedding API batcher
│   │   ├── upserter.py             # Qdrant upsert
│   │   └── tracker.py              # Sync state (which bills are embedded)
│   │
│   └── cache/
│       └── redis_cache.py
│
├── k8s/
│   ├── namespace.yaml
│   ├── qdrant-pv.yaml
│   ├── qdrant-statefulset.yaml
│   ├── qdrant-service.yaml
│   ├── fulltext-pv.yaml
│   ├── nlp-deployment.yaml
│   ├── nlp-service.yaml
│   ├── nlp-sync-cronjob.yaml
│   ├── configmap.yaml
│   └── secrets.yaml                # Template only
│
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    ├── test_fusion.py
    └── eval/
        ├── eval_queries.json       # 100+ queries with expected bills
        └── run_eval.py             # recall@10, MRR
```

---

## 4. Phase-by-Phase Execution

### Phase 0 — Bootstrap (Day 1)

These are one-time manual steps for the `mars` dev cluster before ArgoCD takes over.

```bash
# 1. Prepare dev SSD paths (HostPath storage)
mkdir -p /mnt/data/qdrant
mkdir -p /mnt/data/nlp-fulltext
chown 1000:1000 /mnt/data/qdrant /mnt/data/nlp-fulltext

# 2. Create the standalone csearch-nlp repo
cd ~/Documents/projects/
mkdir csearch-nlp && cd csearch-nlp
git init

# 3. Create read-only PostgreSQL role
# Replace with your actual Postgres superuser connection string or exec into pod:
kubectl --context mars exec -it svc/postgres-service -- psql -U postgres -d csearch -c "
CREATE ROLE csearch_readonly WITH LOGIN PASSWORD 'strong-pass';
GRANT CONNECT ON DATABASE csearch TO csearch_readonly;
GRANT USAGE ON SCHEMA public TO csearch_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO csearch_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO csearch_readonly;
"

# 4. Apply the ArgoCD Application resource (pointing to the new csearch-nlp repo)
kubectl --context mars apply -f argo/applications/csearch-nlp.yaml
argocd app wait csearch-nlp

# 5. Create secrets (not in git)
# For local Ollama dev, the LLM / Embedding API keys can be dummy values
kubectl --context mars create secret generic csearch-nlp-secrets \
  -n csearch-nlp \
  --from-literal=pg-connection-string="postgresql://csearch_readonly:strong-pass@postgres-service.default.svc.cluster.local:5432/csearch" \
  --from-literal=embedding-api-key="ollama-local" \
  --from-literal=llm-api-key="ollama-local"

kubectl --context mars label secret csearch-nlp-secrets \
  -n csearch-nlp \
  argocd.argoproj.io/managed-by=manual

# 6. Verify Qdrant is running and create the collection
kubectl --context mars -n csearch-nlp get pods
kubectl --context mars port-forward svc/qdrant 6333:6333 -n csearch-nlp
```

### Phase 1 — Data pipeline (Week 1–2)

Build the Python pipeline in the new `csearch-nlp` repository, using **local Ollama embeddings (nomic-embed-text or mxbai-embed-large)** and integrating both bills and votes (~10M chunks).

```bash
# 1. Point the pipeline to your local Ollama instance (GPU 3090)
export OLLAMA_HOST="http://localhost:11434"
export EMBEDDING_MODEL="nomic-embed-text"

# 2. Fetch full bill + vote data
python -m csearch_nlp.pipeline fetch --congress-range 93-118 --include-votes

# 3. Chunk the XML & JSON
python -m csearch_nlp.pipeline chunk --congress-range 93-118 --skip-boilerplate --include-votes

# 4. Embed and push to Qdrant using Local GPU
kubectl --context mars port-forward svc/qdrant 6333:6333 -n csearch-nlp &
python -m csearch_nlp.pipeline batch \
  --workers 4 \
  --congress-range 93-118 \
  --batch-size 100 \
  --use-ollama

# 5. Verify
python -c "
from qdrant_client import QdrantClient
c = QdrantClient('localhost', port=6333)
print(c.get_collection('bill_chunks'))
print(c.get_collection('vote_chunks'))
"
# Expect ~10M points
```

### Phase 2 — API service (Week 2–3)

```bash
# 1. Build and push the Docker image
docker build -t 10.0.0.3:30252/csearch-nlp:v0.1.0 .
docker push 10.0.0.3:30252/csearch-nlp:v0.1.0

# 2. Update k8s/nlp-deployment.yaml with the image tag
# 3. Commit and push — ArgoCD deploys automatically

# 4. Test via port-forward
kubectl --context mars port-forward svc/csearch-nlp-api 8000:8000 -n csearch-nlp

curl -X POST http://localhost:8000/api/nlp/search \
  -H "Content-Type: application/json" \
  -d '{"query": "bills about banning stock trading by members of Congress"}'
```

### Phase 3 — Nightly sync (Week 3)

The CronJob is already deployed by ArgoCD (from `k8s/nlp-sync-cronjob.yaml`). Test it manually:

```bash
kubectl --context mars create job --from=cronjob/csearch-nlp-sync test-sync -n csearch-nlp
kubectl --context mars -n csearch-nlp logs job/test-sync -f
```

### Phase 4 — Frontend integration (Week 4)

The API is accessible within the cluster at `csearch-nlp-api.csearch-nlp.svc.cluster.local:8000`. Wire it into the CSearch Nuxt frontend or have the Fastify backend proxy to it.
