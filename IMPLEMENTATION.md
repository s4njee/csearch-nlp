# CSearch NLP Implementation

Architecture: embeddings live in PostgreSQL via the `pgvector` extension. The ingestion pipeline (`project-tarp/`) is operational. This document covers the schema, query patterns, and remaining API work.

## 1. PostgreSQL as the Vector Store

### Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This must be enabled in the target PostgreSQL database before any embedding tables are created.

### Schema Layout

Recommended schema:

```sql
CREATE SCHEMA IF NOT EXISTS nlp;
```

Recommended tables:

```sql
CREATE TABLE nlp.bill_chunks (
  id BIGSERIAL PRIMARY KEY,
  bill_id TEXT NOT NULL,
  congress INTEGER NOT NULL,
  bill_type TEXT NOT NULL,
  bill_number TEXT NOT NULL,
  chunk_type TEXT NOT NULL,
  section_path TEXT,
  title TEXT,
  body TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  source_version TEXT,
  source_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE nlp.bill_embeddings (
  chunk_id BIGINT PRIMARY KEY REFERENCES nlp.bill_chunks(id) ON DELETE CASCADE,
  embedding vector(1536) NOT NULL,
  model TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE nlp.sync_state (
  bill_id TEXT PRIMARY KEY,
  source_hash TEXT NOT NULL,
  source_updated_at TIMESTAMPTZ,
  last_chunked_at TIMESTAMPTZ,
  last_embedded_at TIMESTAMPTZ,
  last_loaded_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  error_text TEXT
);
```

`embedding vector(1536)` matches OpenAI `text-embedding-3-small`, which is the current embedding provider. Do not mix dimensions in one table.

Recommended supporting indexes:

```sql
CREATE INDEX bill_chunks_bill_id_idx ON nlp.bill_chunks (bill_id);
CREATE INDEX bill_chunks_congress_idx ON nlp.bill_chunks (congress);
CREATE INDEX bill_chunks_bill_type_idx ON nlp.bill_chunks (bill_type);
CREATE INDEX bill_chunks_chunk_type_idx ON nlp.bill_chunks (chunk_type);
CREATE INDEX bill_chunks_source_hash_idx ON nlp.bill_chunks (source_hash);
```

### Vector Index

Start with `hnsw` unless the dataset is too large for index build behavior or operational constraints:

```sql
CREATE INDEX bill_embeddings_embedding_hnsw_idx
ON nlp.bill_embeddings
USING hnsw (embedding vector_cosine_ops);
```

If build time or memory pressure becomes a problem, fall back to `ivfflat`:

```sql
CREATE INDEX bill_embeddings_embedding_ivfflat_idx
ON nlp.bill_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Use one ANN strategy first. Do not maintain both unless benchmarking proves you need them.

## 2. Query Pattern

The basic semantic retrieval query should look like this:

```sql
SELECT
  c.id,
  c.bill_id,
  c.chunk_type,
  c.title,
  c.body,
  1 - (e.embedding <=> $1::vector) AS similarity
FROM nlp.bill_embeddings e
JOIN nlp.bill_chunks c ON c.id = e.chunk_id
WHERE c.congress BETWEEN $2 AND $3
ORDER BY e.embedding <=> $1::vector
LIMIT 40;
```

Important points:

- Use cosine distance consistently at ingest and query time.
- Apply structured filters before `LIMIT` where possible.
- Keep chunk metadata on the chunk row, not inside a JSON payload, so SQL filters stay simple and indexable.

## 3. Semantic Search API Route

This section describes the single concrete route Codex should implement first. The goal is a FastAPI endpoint that embeds an incoming query with OpenAI and returns the 20 most similar bills from `nlp.bill_embeddings`.

### Endpoint

```
POST /api/search/semantic
```

### Request model

```python
class SemanticSearchRequest(BaseModel):
    query: str
    congress_min: int | None = None   # optional congress filter
    congress_max: int | None = None
```

### Response model

Return one record per **bill** (not per chunk). Where a bill produces multiple chunk hits, keep only the highest-scoring chunk.

```python
class BillResult(BaseModel):
    bill_id: str
    congress: int
    bill_type: str
    bill_number: str
    title: str | None
    body: str           # the best-matching chunk body, for context
    chunk_type: str
    similarity: float

class SemanticSearchResponse(BaseModel):
    results: list[BillResult]
```

### Implementation steps

**1. Embed the query with OpenAI**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()   # reads OPENAI_API_KEY from env

async def embed_query(text: str) -> list[float]:
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding
```

**2. Query pgvector**

Use cosine distance (`<=>`). Return the top 40 chunk candidates, then deduplicate to 20 unique bills in Python — this is simpler than a `DISTINCT ON` approach and lets you see the best chunk per bill.

```sql
SELECT
  c.bill_id,
  c.congress,
  c.bill_type,
  c.bill_number,
  c.title,
  c.body,
  c.chunk_type,
  1 - (e.embedding <=> $1::vector) AS similarity
FROM nlp.bill_embeddings e
JOIN nlp.bill_chunks c ON c.id = e.chunk_id
WHERE
  ($2::int IS NULL OR c.congress >= $2)
  AND ($3::int IS NULL OR c.congress <= $3)
ORDER BY e.embedding <=> $1::vector
LIMIT 40;
```

Parameters: `$1` = embedding vector, `$2` = `congress_min`, `$3` = `congress_max`.

**3. Deduplicate to 20 bills**

```python
seen: dict[str, BillResult] = {}
for row in rows:
    if row["bill_id"] not in seen:
        seen[row["bill_id"]] = BillResult(**row)
    if len(seen) == 20:
        break
results = list(seen.values())
```

**4. Wire up the route**

```python
@router.post("/api/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(body: SemanticSearchRequest, db: AsyncConnection = Depends(get_db)):
    vector = await embed_query(body.query)
    rows = await db.fetch(SEMANTIC_SEARCH_SQL, vector, body.congress_min, body.congress_max)
    results = deduplicate_to_bills(rows, limit=20)
    return SemanticSearchResponse(results=results)
```

### Notes for Codex

- The embedding column is `vector(1536)` — `text-embedding-3-small` output. Do not change the dimension.
- Use `asyncpg` for the database connection; it handles `vector` columns as plain Python lists of floats when passed as parameters. Cast explicitly in the SQL (`$1::vector`) to be safe.
- `OPENAI_API_KEY` is injected from the `csearch-nlp-secrets` Secret in k8s; read it from env — do not hardcode.
- Keep the embedding call and the DB query independent so each can be tested separately.
- Do not add reranking or hybrid fusion to this route yet — that is section 4 (Hybrid Retrieval) and comes later.

## 4. Hybrid Retrieval

The retrieval service should execute two searches in parallel:

### Vector Search

- embed query with the same model used at ingest time
- search `nlp.bill_embeddings`
- return top `k` chunk candidates plus similarity scores

### Keyword Search

- search existing CSearch bill metadata with `tsvector`
- return top keyword-matched bills

### Fusion

- convert both result sets into ranked lists
- merge with Reciprocal Rank Fusion
- rerank the fused top candidates with a cross-encoder

The final user-facing unit can still be a bill, even though the vector retrieval unit is a chunk.

## 5. Ingestion Pipeline

The current proof-of-concept scripts in `project-tarp/` already cover fetch, chunk, and embedding well enough to inform the production pipeline. The change required is at the load stage.

### Fetch

- download bill text from GovInfo
- keep a local raw cache
- prefer the highest-value available bill version

### Chunk

- preserve section boundaries where possible
- strip boilerplate
- deduplicate exact repeated text before embedding
- emit stable chunk identifiers derived from bill/version/section content

### Embed

- batch through OpenAI `text-embedding-3-small` consistently for both corpus and queries
- checkpoint progress by shard or batch, not one monolithic JSON file
- write intermediate outputs to local disk only as long as needed for restart safety

### Load

Load in two phases:

1. bulk insert `nlp.bill_chunks`
2. bulk insert `nlp.bill_embeddings`

Preferred approach:

- stage rows into temp tables or unlogged tables
- merge into target tables in SQL
- update `nlp.sync_state` only after successful commit

Do not insert row-by-row through the application if you can avoid it. Use PostgreSQL bulk loading.

## 6. Suggested Loader Workflow

Recommended transactional flow for one bill:

1. compute source hash
2. compare against `nlp.sync_state`
3. if unchanged, skip
4. if changed, delete existing rows for that bill
5. insert fresh chunk rows
6. insert fresh embedding rows
7. update sync state
8. commit

This is simpler and safer than attempting fine-grained partial mutation on the first version.

## 7. API Configuration

Example `ConfigMap` values:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: csearch-nlp-config
  namespace: csearch-nlp
data:
  PG_CONNECTION_STRING: "postgresql://csearch_readonly:password@postgres-service.default.svc.cluster.local:5432/csearch"
  EMBEDDING_MODEL: "text-embedding-3-small"
  EMBEDDING_DIMENSIONS: "1536"
  RAG_VECTOR_TOP_K: "40"
```

Example secret values:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: csearch-nlp-secrets
  namespace: csearch-nlp
type: Opaque
stringData:
  pg-connection-string: "postgresql://csearch_nlp_writer:strong-pass@postgres-service.default.svc.cluster.local:5432/csearch"
  openai-api-key: "sk-proj-..."
```

## 8. Kubernetes Footprint

Required:

- namespace (`csearch-nlp`)
- API deployment
- secrets/configmaps
- PVC for working data (tarp-data-pvc, congress-data-pvc)
- CronJob for nightly sync (`tarp-nightly-updater`)

## 10. Nightly Sync

The nightly updater is deployed as a Kubernetes CronJob (`tarp-nightly-updater`) and orchestrated by `project-tarp/nightly_update.sh`. Each step is idempotent — re-running converges cleanly without manual cleanup.

Pipeline order: `fetcher.py` → `content_hasher.py` → `chunker.py` → `embedder.py` → `upserter.py`

See `UPDATE.md` for full operational details.

## 11. Operational Notes

### PostgreSQL Tuning

Moving vectors into PostgreSQL makes database tuning more important than before. At minimum, evaluate:

- `shared_buffers`
- `work_mem`
- `maintenance_work_mem`
- autovacuum thresholds on `nlp` tables
- index build timing and concurrency

### Re-Embedding Strategy

Because embeddings are isolated in `nlp.bill_embeddings`, model migrations are straightforward:

- keep chunk text stable
- create a new embeddings table or a new `model` partition
- backfill vectors
- switch retrieval to the new model once validated

Do not overwrite the only embeddings copy during a model migration.

### Backup Strategy

Backups now come from PostgreSQL backups rather than a separate Qdrant snapshot flow. That is operationally simpler, but it also means the database backup policy now covers both transactional and vector data.

## 12. Rollout Sequence

Steps 1–4 are complete. Remaining:

5. wire up semantic search API route (section 3)
6. wire up hybrid retrieval (section 4)
7. add reranking and answer generation

## 13. Files and Modules

Suggested project layout after the migration:

```text
csearch_nlp/
  api/
    server.py
    routes.py
    models.py
  rag/
    query_classifier.py
    embedder.py
    retriever.py
    fusion.py
    reranker.py
    hydrator.py
    prompt_builder.py
    generator.py
  pipeline/
    fetcher.py
    chunker.py
    embedder.py
    loader.py
    tracker.py
    sync.py
  db/
    migrations/
    queries/
  cache/
    redis_cache.py
```

The key difference from the older design is `loader.py` and SQL migrations replace the Qdrant upsert layer.

