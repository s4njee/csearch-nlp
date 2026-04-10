# Upserting Embeddings into PostgreSQL + Migrating to VPS

---

## Dockerized Upserter

If you want to run the loader from a freestanding container on Freya, build the
image from `project-tarp/` and mount the extracted shard directory into the
container at `/home/sanjee/nlp/embedded_chunks`:

```bash
docker build -f project-tarp/Dockerfile.upserter -t csearch-upserter project-tarp

docker run --rm \
  -e PG_CONNECTION_STRING="postgresql://USER:PASSWORD@POSTGRES_HOST:5432/csearch" \
  -v /home/sanjee/nlp/embedded_chunks:/home/sanjee/nlp/embedded_chunks:ro \
  csearch-upserter --recreate --batch-size 2000
```

This assumes the target database already exists and the `nlp` schema is live
alongside the bill summary tables. If you need to resume a partial load, drop
`--recreate` and add `--skip-hnsw` until the final index build run.

Resume example:

```bash
docker run --rm \
  -e PG_CONNECTION_STRING="postgresql://USER:PASSWORD@POSTGRES_HOST:5432/csearch" \
  -v /home/sanjee/nlp/embedded_chunks:/home/sanjee/nlp/embedded_chunks:ro \
  csearch-upserter --batch-size 2000 --skip-hnsw
```

Once all shards are loaded successfully, run an index-only pass:

```bash
docker run --rm \
  -e PG_CONNECTION_STRING="postgresql://USER:PASSWORD@POSTGRES_HOST:5432/csearch" \
  -v /home/sanjee/nlp/embedded_chunks:/home/sanjee/nlp/embedded_chunks:ro \
  csearch-upserter --index-only
```

If PostgreSQL is running in a smaller container or k8s pod, lower the HNSW
build memory knobs:

```bash
docker run --rm \
  -e PG_CONNECTION_STRING="postgresql://USER:PASSWORD@POSTGRES_HOST:5432/csearch" \
  -v /home/sanjee/nlp/embedded_chunks:/home/sanjee/nlp/embedded_chunks:ro \
  csearch-upserter --index-only \
    --maintenance-work-mem 256MB \
    --max-parallel-maintenance-workers 1
```

### k3s Import + Job

If you want to load the image into the Freya k3s container runtime instead of
pulling it from a registry, save and import it with `ctr`:

```bash
docker build -f project-tarp/Dockerfile.upserter -t csearch-upserter:latest project-tarp
docker save csearch-upserter:latest | ssh freya sudo ctr -n k8s.io images import -
```

Then apply the Kubernetes Job manifest:

```bash
kubectl --context freya apply -f k8s/secrets.yaml
kubectl --context freya apply -f k8s/nlp-upserter-job.yaml
```

That job:

- mounts `/home/sanjee/nlp/embedded_chunks` from the Freya node with `hostPath`
- targets the `postgres` service in the `default` namespace
- starts in resume mode with `--skip-hnsw`

When all shards are loaded, apply the final-index job:

```bash
kubectl --context freya apply -f k8s/nlp-upserter-final-job.yaml
```

## Step 1 — Install pgvector

```bash
# macOS (Homebrew)
brew install pgvector

# Ubuntu/Debian
sudo apt install postgresql-16-pgvector
```

`uv` handles the Python dependency (`psycopg2-binary`) automatically via the inline script header.

---

## Step 2 — Tune PostgreSQL for Bulk Load

Edit `postgresql.conf` (find it with `SHOW config_file;` in psql) before running the load:

```ini
shared_buffers = 8GB
work_mem = 256MB
maintenance_work_mem = 16GB
max_parallel_maintenance_workers = 6
max_parallel_workers = 8
wal_buffers = 256MB
checkpoint_completion_target = 0.9
synchronous_commit = off
autovacuum = off
```

Restart Postgres after editing:

```bash
brew services restart postgresql@16   # macOS
sudo systemctl restart postgresql     # Linux
```

Set your connection string:

```bash
export PG_CONNECTION_STRING="postgresql://$(whoami)@localhost:5432/csearch"
```

---

## Step 3 — Dry Run

Validate all shards are present and parseable before touching the database:

```bash
cd project-tarp
uv run upserter.py --dry-run
```

Check the log output — it should report the number of shards and total chunk count (~2.8M chunks).

---

## Steps 4–6 — Load, Index, Verify

The `csearch` database already exists alongside the bill summary tables. Just enable pgvector in it, then run the upserter:

```bash
psql "$PG_CONNECTION_STRING" -c "CREATE EXTENSION IF NOT EXISTS vector;"

uv run upserter.py --recreate --batch-size 2000
```

What it does in order:
1. Creates `nlp.bill_chunks` and `nlp.bill_embeddings` tables (drops existing if `--recreate`)
2. Loads all shards transactionally
3. Builds the HNSW index (`m=16, ef_construction=128`) in one bulk pass after all rows are loaded
4. Verifies and logs final row counts

**Resume after interruption** (drop `--recreate`, add `--skip-hnsw` until the final run):

```bash
# Resume loading without wiping the tables:
uv run upserter.py --batch-size 2000 --skip-hnsw

# Once all shards are loaded, build only the index:
uv run upserter.py --index-only

# On smaller Postgres instances, lower HNSW build memory:
uv run upserter.py --index-only \
  --maintenance-work-mem 256MB \
  --max-parallel-maintenance-workers 1
```

**Tune HNSW parameters** (optional — higher values improve recall at the cost of build time):

```bash
uv run upserter.py --recreate --batch-size 2000 \
  --hnsw-m 24 --hnsw-ef-construction 200
```

Re-enable autovacuum after the load completes:

```sql
psql "$PG_CONNECTION_STRING" -c "ALTER SYSTEM SET autovacuum = on; SELECT pg_reload_conf();"
```

---

## Step 7 — Dump the nlp Schema

Only dump the `nlp` schema — no need to touch the rest of the `csearch` database.

```bash
pg_dump \
  --format=custom \
  --compress=9 \
  --no-acl \
  --no-owner \
  --schema=nlp \
  --file=csearch_nlp.dump \
  "$PG_CONNECTION_STRING"
```

Expect 35–50 GB uncompressed, significantly less with `--compress=9`.

---

## Step 8 — Transfer to VPS

```bash
rsync -avz --progress csearch_nlp.dump user@your-vps:/home/user/
```

---

## Step 9 — Restore on the VPS

The `csearch` database already exists on the VPS. Enable pgvector and restore just the `nlp` schema into it:

```bash
psql csearch -c "CREATE EXTENSION IF NOT EXISTS vector;"

pg_restore \
  --dbname=csearch \
  --no-acl \
  --no-owner \
  --jobs=4 \
  csearch_nlp.dump
```

Verify counts after restore:

```bash
psql csearch -c "
SELECT
  (SELECT count(*) FROM nlp.bill_chunks)     AS chunks,
  (SELECT count(*) FROM nlp.bill_embeddings) AS embeddings;
"
```

---

## Step 10 — VPS Query Tuning

Set in `postgresql.conf` on the VPS:

```ini
shared_buffers = 4GB
work_mem = 64MB
hnsw.ef_search = 100     # higher = better recall, slower queries (default 40)
autovacuum = on
```

Confirm the index is being used:

```sql
EXPLAIN ANALYZE
SELECT c.bill_id, c.body, e.embedding <=> '[0.1, 0.2, ...]'::vector AS distance
FROM nlp.bill_embeddings e
JOIN nlp.bill_chunks c ON c.id = e.chunk_id
ORDER BY distance
LIMIT 10;
```

The query plan should show `Index Scan using bill_embeddings_embedding_hnsw_idx`.
