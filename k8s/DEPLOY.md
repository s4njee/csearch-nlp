# Kubernetes Deployment Guide for csearch-nlp

This document overviews the Kubernetes deployment components for the `csearch-nlp` project. All components are designed to be deployed into a dedicated namespace to maintain isolation and manageability.

## Architecture Overview

The deployment consists of three main parts:
1. **Core Configuration & Base Resources**: Namespace, ConfigMaps, and Secrets.
2. **Qdrant Vector Database**: A StatefulSet deployment of Qdrant with its requisite persistence and services.
3. **NLP Application**: The main API deployment and a scheduled CronJob for daily data synchronization.

Some components are specifically scheduled on the `worker1` node (via `nodeSelector`) to optimize spatial locality or satisfy GPU/Hardware requirements for local model inference.

---

## 1. Core Configuration

### `namespace.yaml`
- **Purpose**: Defines the `csearch-nlp` namespace.
- **Details**: All subsequent resources are deployed into this namespace, keeping them isolated from the rest of the cluster.

### `secrets.yaml`
- **Purpose**: Stores sensitive information securely using Kubernetes `Secret` type `Opaque`.
- **Details**:
  - `pg-connection-string`: Connection URL for your Postgres database (`csearch`).
  - `embedding-api-key`: API key for the embedding model (e.g., `ollama-local`).
  - `llm-api-key`: API key for the LLM model (e.g., `ollama-local`).

### `configmap.yaml`
- **Purpose**: Manages non-sensitive environment variables and configuration data for the NLP components.
- **Details**:
  - Contains Qdrant connection specifics (`QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`).
  - Configures model choices (`EMBEDDING_MODEL: nomic-embed-text`, `LLM_MODEL: qwen2.5:7b`).
  - Sets RAG behavior and thresholds (`RAG_VECTOR_TOP_K`, `RRF_K`, `RAG_SCORE_THRESHOLD`, etc.).
  - Configures Redis cache details, TTL values, and GovInfo API rate limits.

---

## 2. Qdrant Vector Data Store

### `qdrant-pv.yaml`
- **Purpose**: Configures persistence for the vector database.
- **Details**: Defines a 50Gi PersistentVolume (PV) and PersistentVolumeClaim (PVC). By default, it uses a `hostPath` at `/mnt/data/qdrant` and standard storage class with a `ReadWriteOnce` access mode.

### `qdrant-statefulset.yaml`
- **Purpose**: Deploys the Qdrant instance.
- **Details**:
  - Uses the `qdrant/qdrant:v1.12.1` image.
  - Deployed as a `StatefulSet` with 1 replica to ensure data safety across pod restarts.
  - Mounts the `qdrant-pvc` claim to `/qdrant/storage`.
  - Exposes REST (6333) and gRPC (6334) ports.
  - Schedules strictly on node `worker1`.

### `qdrant-service.yaml`
- **Purpose**: Networking component for Qdrant.
- **Details**: A `ClusterIP` service exposing Qdrant’s REST (6333) and gRPC (6334) ports internally across the cluster, allowing the local NLP API to resolve Qdrant via `qdrant.csearch-nlp.svc.cluster.local`.

---

## 3. NLP Application

### `fulltext-pv.yaml`
- **Purpose**: Configures shared persistence for caching parsed bill full-texts.
- **Details**: Defines a 20Gi `ReadWriteMany` PV and PVC mapped to a `hostPath` at `/mnt/data/nlp-fulltext`. This allows multiple pods (API and sync cron job) to share and read the downloaded bill textual data simultaneously.

### `nlp-deployment.yaml`
- **Purpose**: Deploys the main `csearch-nlp-api` web service.
- **Details**:
  - Pulls the main application image (`10.0.0.3:30252/csearch-nlp:latest`).
  - Runs exactly 1 replica, scheduled on the `worker1` node to ensure hardware locality.
  - Automatically imports environment variables from both `csearch-nlp-config` and `csearch-nlp-secrets`.
  - Mounts the `fulltext-cache` volume at `/data/fulltext`.
  - Configured with memory and cpu constraints, plus liveness & readiness probes.

### `nlp-service.yaml`
- **Purpose**: Networking interface for the application API.
- **Details**: Replicating a simple `ClusterIP` service to make the NLP API reachable on port 8000 within the cluster.

### `nlp-sync-cronjob.yaml`
- **Purpose**: Automates periodic data syncing chores.
- **Details**: 
  - Set up as a `CronJob` to run daily at 00:30 UTC (`30 0 * * *`).
  - Launches the container running `python -m csearch_nlp.pipeline sync` to keep vector data and full-text caches up to date.
  - Shares the same secrets, configmap, and persistent volume (`fulltext-pvc`) as the API deployment.
  - Scheduled to explicitly run on the `worker1` node.

---

## Deployment Instructions

To spin up the entire application stack across the cluster, run the following commands in sequence:

1. **Create the Namespace First**:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   ```

2. **Apply Configurations and Secrets**:
   Before running this, make sure you configure your database and API keys properly in `secrets.yaml`.
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   ```

3. **Deploy the Vector Database & Cache Volumes**:
   ```bash
   kubectl apply -f k8s/qdrant-pv.yaml
   kubectl apply -f k8s/qdrant-service.yaml
   kubectl apply -f k8s/qdrant-statefulset.yaml
   kubectl apply -f k8s/fulltext-pv.yaml
   ```

4. **Deploy the NLP Application & CronJob**:
   ```bash
   kubectl apply -f k8s/nlp-deployment.yaml
   kubectl apply -f k8s/nlp-service.yaml
   kubectl apply -f k8s/nlp-sync-cronjob.yaml
   ```

To verify that applications are running as expected:
```bash
kubectl get pods -n csearch-nlp
```
