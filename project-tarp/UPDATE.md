# Nightly Bill Update Pipeline

Incremental pipeline that picks up new bills from the `@unitedstates/congress`
scraper, fetches their text from GovInfo, chunks, embeds, and upserts into
PostgreSQL. Runs as a Kubernetes CronJob on freya.

---

## How Each Step Handles Incremental Runs

| Script | Incremental behaviour |
|---|---|
| `fetcher.py` | Skips any bill where `.meta.json` already exists — only new bills are downloaded |
| `content_hasher.py` | Hashes legislative text (XML attributes stripped), compares to stored manifest — exits early if nothing meaningfully changed |
| `chunker.py` | Full rewrite per congress — fast, CPU-only (~minutes) |
| `embedder.py` | Skips chunks whose identity (bill_id + content hash + chunk index) already exists — only new chunks cost API money |
| `upserter.py` | Idempotent per `bill_id` — deletes and reinserts changed rows only. `--skip-hnsw` leaves the HNSW index to pgvector's incremental updates |

### Why content hashing?

The `@unitedstates/congress` scraper sometimes refreshes bill XMLs where only
metadata has changed — effective dates, action dates, print numbers in XML
attributes — while the legislative text is identical. Without hashing,
this would trigger unnecessary chunking, embedding checks, and upserts.

`content_hasher.py` extracts only text nodes from the XML (stripping all
attributes), hashes each bill, and compares against the manifest from the
previous run. If nothing has changed, the pipeline exits before chunking.

---

## Repository Layout

```
project-tarp/
  fetcher.py               # downloads bill text from GovInfo
  chunker.py               # splits bills into chunks
  embedder.py              # generates embeddings via OpenAI
  upserter.py              # loads chunks + vectors into PostgreSQL
  content_hasher.py        # detects meaningful text changes between runs
  nightly_update.sh        # orchestrates the pipeline
  Dockerfile.nightly-updater

k8s/
  tarp-data-pv.yaml                      # PV/PVC for TARP working data + congress scraper data
  nlp-nightly-updater-sealed-secret.yaml # SealedSecret for OpenAI API key
  nlp-nightly-updater-cronjob.yaml       # CronJob definition
```

---

## Building the Container

Build from the repo root and push to the local registry. The updater image uses
the existing amd64 `csearch-upserter:latest` image as its base, so make sure that
image exists locally before building:

```bash
DOCKER_BUILDKIT=0 docker build --platform linux/amd64 \
  -f project-tarp/Dockerfile.nightly-updater \
  -t registry.s8njee.com/csearch-tarp-updater:latest \
  project-tarp

docker push registry.s8njee.com/csearch-tarp-updater:latest
```

Rebuild and push after any script change before the next scheduled run.

---

## Sealed Secret for the OpenAI API Key

The CronJob currently reads `OPENAI_API_KEY` from the existing
`csearch-nlp-secrets` Secret. For a commit-safe setup, move that key to a
SealedSecret encrypted with the cluster's public key.

### One-time setup

```bash
# Install kubeseal
brew install kubeseal   # macOS
# or download from https://github.com/bitnami-labs/sealed-secrets/releases

# Fetch the cluster's public certificate
kubeseal --fetch-cert \
  --controller-name=sealed-secrets \
  --controller-namespace=kube-system \
  > pub-sealed-secrets.pem
```

### Sealing the key

```bash
# Create the plain Secret (do NOT commit this file)
kubectl create secret generic tarp-updater-secrets \
  --namespace csearch-nlp \
  --from-literal=openai-api-key="sk-proj-..." \
  --dry-run=client -o yaml > /tmp/tarp-secret.yaml

# Seal it
kubeseal \
  --cert pub-sealed-secrets.pem \
  --format yaml \
  < /tmp/tarp-secret.yaml \
  > k8s/nlp-nightly-updater-sealed-secret.yaml

# Delete the plain manifest
rm /tmp/tarp-secret.yaml

# Apply
kubectl apply -f k8s/nlp-nightly-updater-sealed-secret.yaml
```

The controller decrypts it and creates a regular Secret named
`tarp-updater-secrets` in the `csearch-nlp` namespace.

To rotate the key: repeat the steps above with the new key and re-apply.

---

## PostgreSQL Connection String

The Postgres DSN is read from the existing `csearch-nlp-secrets` Secret:

```yaml
- name: PG_CONNECTION_STRING
  valueFrom:
    secretKeyRef:
      name: csearch-nlp-secrets
      key: pg-connection-string
```

---

## Deploying to Kubernetes

### 1. Create the PersistentVolumes

```bash
kubectl apply -f k8s/tarp-data-pv.yaml
```

This creates:
- `tarp-data-pv` / `tarp-data-pvc` — TARP working data on freya at `/home/sanjee/nlp/tarp-data`
- `congress-data-pv` / `congress-data-pvc` — read-only view of `/home/sanjee/congress/data`

Ensure `/home/sanjee/nlp/tarp-data` exists and is writable by uid 1000 on freya:

```bash
ssh freya.local "mkdir -p /home/sanjee/nlp/tarp-data && chown 1000:1000 /home/sanjee/nlp/tarp-data"
```

### 2. Apply the SealedSecret

```bash
kubectl apply -f k8s/nlp-nightly-updater-sealed-secret.yaml
```

Verify it was decrypted:

```bash
kubectl get secret tarp-updater-secrets -n csearch-nlp
```

### 3. Deploy the CronJob

```bash
kubectl apply -f k8s/nlp-nightly-updater-cronjob.yaml
```

Verify:

```bash
kubectl get cronjob tarp-nightly-updater -n csearch-nlp
```

### 4. Trigger a manual run

```bash
kubectl create job --from=cronjob/tarp-nightly-updater tarp-manual-$(date +%s) \
  -n csearch-nlp
```

Watch logs:

```bash
kubectl logs -f -l job-name=tarp-manual-... -n csearch-nlp
```

---

## HNSW Index Maintenance

The nightly run passes `--skip-hnsw` — pgvector updates the HNSW index
incrementally as rows are inserted. At nightly addition rates (~50–500 bills)
relative to a 2.8M-vector corpus, the quality impact is negligible and no
routine rebuild is needed.

If you ever notice measurable recall degradation (validate with `EXPLAIN ANALYZE`
and benchmark queries), you can force a full rebuild:

```bash
python upserter.py --index-only
```

---

## Monitoring

Check how many new rows landed each night:

```sql
SELECT
  congress,
  count(*)               AS chunks,
  max(created_at)        AS last_inserted
FROM nlp.bill_chunks
GROUP BY congress
ORDER BY congress DESC;
```

Check recent job history:

```bash
kubectl get jobs -n csearch-nlp --sort-by=.metadata.creationTimestamp
kubectl logs -n csearch-nlp job/<job-name>
```

Logs are also written to `/home/sanjee/nlp/tarp-data/logs/update-YYYY-MM-DD.log` on freya.

---

## Cost Estimate

`text-embedding-3-small` at $0.02/1M tokens.

| New bills/night | Est. tokens | Est. cost |
|---|---|---|
| 0 (no changes) | 0 | $0.00 |
| 50 | 250K | ~$0.005 |
| 200 | 1M | ~$0.02 |
| 500 | 2.5M | ~$0.05 |

Congress is not in session every day — most nights the content hasher will
detect no changes and the pipeline exits before any API calls are made.
