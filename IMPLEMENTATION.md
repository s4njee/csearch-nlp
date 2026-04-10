# CSearch NLP Implementation

This document translates the project plan into concrete infrastructure, schema, and rollout steps for a `pgvector`-backed implementation.

The main architectural change is simple: embeddings are no longer stored in Qdrant. They live in PostgreSQL via the `vector` extension.

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

`embedding vector(1536)` is correct for OpenAI `text-embedding-3-small`. If this deployment switches to `Qwen3-Embedding-8B`, change the column to match the vector size you actually write:

- `vector(4096)` for the default Qwen3 output
- `vector(1024)` or another reduced size if you intentionally shorten vectors during local inference

Do not mix dimensions in one table.

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

## 3. Hybrid Retrieval

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

## 4. Ingestion Pipeline

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

- batch through one embedding backend consistently for both corpus and queries
- if using OpenAI, `text-embedding-3-small` is the lower-cost hosted default
- if using a local GPU, `Qwen3-Embedding-8B` is the highest-quality open-weight default
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

## 5. Suggested Loader Workflow

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

## 6. API Configuration

The application config should move away from Qdrant-specific variables.

Example `ConfigMap` values:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: csearch-nlp-config
  namespace: csearch-nlp
data:
  PG_CONNECTION_STRING: "postgresql://csearch_readonly:password@postgres-service.default.svc.cluster.local:5432/csearch"
  EMBEDDING_BACKEND: "ollama"
  EMBEDDING_MODEL: "qwen3-embedding:8b-q8_0"
  EMBEDDING_BASE_URL: "http://ollama.default.svc.cluster.local:11434"
  EMBEDDING_DIMENSIONS: "4096"
  EMBEDDING_QUERY_PROMPT: "Represent this query for retrieving relevant legislative passages:"
  LLM_MODEL: "gpt-5.4-nano"
  RAG_VECTOR_TOP_K: "40"
  RAG_RERANK_TOP_K: "10"
  RAG_SCORE_THRESHOLD: "0.35"
  RRF_K: "60"
  REDIS_URL: "redis://redis-service.default.svc.cluster.local:6379/1"
  CACHE_EMBEDDING_TTL: "86400"
  CACHE_SEARCH_TTL: "3600"
  CACHE_LLM_TTL: "3600"
  GOVINFO_RATE_LIMIT: "10"
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
  openai-api-key: "replace-me-if-using-openai"
```

If OpenAI remains the embedding provider, change the embedding settings accordingly:

```yaml
data:
  EMBEDDING_BACKEND: "openai"
  EMBEDDING_MODEL: "text-embedding-3-small"
  EMBEDDING_DIMENSIONS: "1536"
  EMBEDDING_BASE_URL: ""
  EMBEDDING_QUERY_PROMPT: ""
```

## 7. Local Embedding on an RTX 3090

An RTX 3090 is a reasonable target for local `Qwen3-Embedding-8B` inference. For this project, the most practical local options are:

- `Ollama` for the quickest operational path
- `sentence-transformers` or `vLLM` if you need tighter control over prompts, batching, or output dimensions

Recommended default for a 3090:

- start with `qwen3-embedding:8b-q8_0` in Ollama
- keep the PostgreSQL column at `vector(4096)` unless storage pressure forces you to shorten vectors later
- use cosine similarity only
- embed document chunks without a query prefix
- embed search queries with a retrieval-oriented prefix

### Option A: Ollama

Install and start Ollama on the GPU host, then pull the model:

```bash
ollama pull qwen3-embedding:8b-q8_0
```

Smoke test the local daemon:

```bash
curl http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding:8b-q8_0",
    "input": "Clean Water Act section 402 permitting requirements"
  }'
```

Batch embedding example in Python:

```python
import ollama

QUERY_PREFIX = "Represent this query for retrieving relevant legislative passages: "

def embed_documents(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(
        model="qwen3-embedding:8b-q8_0",
        input=texts,
    )
    return response["embeddings"]

def embed_queries(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(
        model="qwen3-embedding:8b-q8_0",
        input=[QUERY_PREFIX + text for text in texts],
    )
    return response["embeddings"]
```

Operational notes for Ollama:

- verify the returned vector length once and make the PostgreSQL column match it exactly
- keep the same model tag for both indexing and query embedding
- if you later change quantization or dimensions, rebuild the embeddings table rather than mixing vectors

### Option B: Direct Hugging Face Inference

Use this path if you want the official prompt handling and optional dimension shortening in one process.

Install the minimum packages:

```bash
pip install "transformers>=4.51.0" "sentence-transformers>=2.7.0" torch
```

Example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
)

doc_embeddings = model.encode(
    document_texts,
    normalize_embeddings=True,
)

query_embeddings = model.encode(
    query_texts,
    prompt_name="query",
    normalize_embeddings=True,
)
```

This path is easier to extend if you later need custom prompt handling, larger batches, or your own vector post-processing before writing into `pgvector`.

### Pipeline Changes Required

To switch the project from hosted OpenAI embeddings to local Qwen3 embeddings:

1. update the schema so `nlp.bill_embeddings.embedding` matches the chosen Qwen3 dimension
2. replace the embedding client in both ingest and query paths with one local backend
3. apply the same query prefix strategy every time a search query is embedded
4. re-embed the full corpus once, then rebuild the ANN index
5. keep the reranker and answer-generation path independent from the embedding provider

## 8. Kubernetes Footprint

With `pgvector`, the deployment is smaller than the old Qdrant plan.

You still need:

- namespace
- API deployment
- Redis connectivity
- secrets/configmaps
- optional PVC for cached raw full text
- optional CronJob for nightly sync

You do not need:

- Qdrant StatefulSet
- Qdrant service
- Qdrant PV/PVC
- Qdrant collection bootstrap step

That simplification is the main operational win of the migration.

## 9. Nightly Sync

The sync job should:

1. query for new or updated bills
2. fetch current source text
3. re-chunk changed bills
4. embed changed chunks
5. replace rows in PostgreSQL
6. update sync state

Example CronJob command:

```yaml
args:
  - python
  - -m
  - csearch_nlp.pipeline.sync
```

Keep the job idempotent. If it fails halfway through, re-running it should converge cleanly without manual cleanup.

## 10. Operational Notes

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

## 11. Rollout Sequence

Recommended rollout order:

1. enable `pgvector` in a development database
2. create `nlp` schema and tables
3. load a small sample corpus from `project-tarp/`
4. benchmark vector search latency and recall
5. wire up hybrid retrieval in the API
6. add reranking and answer generation
7. add scheduled incremental sync
8. only then scale corpus size

## 12. Files and Modules

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

## 13. Current Implication for Existing POC Code

`project-tarp/` still contains Qdrant-oriented proof-of-concept tooling. That is acceptable as historical prototype code, but it should not be treated as the storage design for the main system anymore.

When promoting code from the prototype:

- keep fetch logic
- keep chunking logic
- keep embedding logic
- replace Qdrant-specific load and query layers with PostgreSQL implementations

That boundary should stay explicit so the repository does not drift into supporting two vector stores by accident.
