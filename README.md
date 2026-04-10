# CSearch NLP

Semantic search and grounded question answering over U.S. Congressional legislation.

Embeddings are generated with OpenAI `text-embedding-3-small` and stored in PostgreSQL using the `pgvector` extension. Answer generation uses `gpt-5.4-nano`. The service is designed to run alongside the existing CSearch stack and reuse PostgreSQL and Redis instead of introducing a separate vector database.

## Why This Exists

CSearch already has strong keyword search with PostgreSQL full-text indexing over bill titles and summaries. That works well for exact terms, but it breaks down on queries like:

- "bills that would make it harder for companies to pollute rivers"
- "legislation protecting gig workers from being classified as independent contractors"
- "what has Congress done about prescription drug prices since 2020"

This project adds a retrieval pipeline that combines:

- PostgreSQL keyword search
- `pgvector` semantic similarity search
- reranking
- LLM answer generation grounded in retrieved bill text

The goal is not to replace keyword search. The goal is to supplement it with a query path that can handle meaning, intent, and paraphrase.

## Architecture

```text
                    ┌─────────────────────────────┐
                    │    CSearch Frontend (Nuxt)  │
                    └──────────────┬──────────────┘
                                   │
                          POST /api/nlp/search
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              csearch-nlp service (FastAPI)                      │
│                                                                  │
│  1. Parse query and optional filters                             │
│  2. Embed query via OpenAI API                                   │
│  3. Run pgvector similarity search                               │
│  4. Run PostgreSQL full-text search in parallel                  │
│  5. Fuse candidate lists with Reciprocal Rank Fusion             │
│  6. Re-rank top hits with a cross-encoder                        │
│  7. Hydrate bill metadata from PostgreSQL                        │
│  8. Stream grounded answer via OpenAI API                        │
│                                                                  │
│  Redis (shared, `nlp:` prefix): embedding + retrieval + LLM cache│
└──────────────┬───────────────────────────┬───────────────────────┘
               │                           │
               ▼                           ▼
       ┌──────────────┐            ┌──────────────┐
       │ PostgreSQL   │            │ OpenAI       │
       │ + pgvector   │            │ embed +      │
       │ + tsvector   │            │ generate     │
       └──────────────┘            └──────────────┘
```

## Storage Model

The previous design used Qdrant. This project now standardizes on `pgvector`.

That changes the system in a few important ways:

- One persistence layer instead of two. Bill metadata, keyword indexes, and vector embeddings all live in PostgreSQL.
- Simpler operations. No separate Qdrant StatefulSet, collection management, snapshot process, or network hop.
- Easier joins. Retrieval can score vector matches and immediately join against bill metadata and chunk records in SQL.
- Slightly different tuning. Instead of Qdrant HNSW collection settings, we manage `pgvector` indexes, row layout, and PostgreSQL memory settings.

At a high level, the core tables are:

- `nlp_bill_chunks`: chunk text and metadata
- `nlp_bill_embeddings`: 1536-dim vectors for each chunk
- `nlp_sync_state`: incremental ingestion bookkeeping

The bill corpus itself remains sourced from the existing CSearch data plus downloaded GovInfo full text.

## Retrieval Strategy

The query-time path uses hybrid retrieval:

1. Embed the normalized user query with `text-embedding-3-small`.
2. Run a `pgvector` nearest-neighbor search against chunk embeddings.
3. Run PostgreSQL full-text search against bill title and summary fields.
4. Fuse both ranked lists with Reciprocal Rank Fusion.
5. Re-rank the top fused results with a cross-encoder.
6. Generate an answer only from retrieved evidence.

This keeps semantic search strong without giving up precise keyword behavior.

## Ingestion Strategy

Embedding every raw XML node across 50 years would produce too many low-value chunks. The ingestion pipeline is selective:

- include titles, summaries, substantive sections, definitions, actions, and sponsor data
- skip boilerplate sections and obsolete duplicate bill versions
- deduplicate exact repeated sections before embedding
- store chunk text separately from embedding rows so re-embedding does not require rewriting the chunk corpus

The current proof-of-concept work in `project-tarp/` remains useful for fetch, chunk, and embed logic. The storage target for the main system, however, is now PostgreSQL with `pgvector`, not Qdrant.

## Implementation Plan

### Step 1: Database and Infra Foundations

Decisions:

- Embedding model: OpenAI `text-embedding-3-small` with 1536 dimensions
- Answer model: OpenAI `gpt-5.4-nano`
- Vector store: PostgreSQL `pgvector`
- Cache: shared Redis with `nlp:` key prefix

What this step covers:

- enable the `vector` extension in PostgreSQL
- create a read-only role for metadata access and a write-capable role for NLP ingestion if needed
- add the NLP schema, chunk tables, embedding tables, indexes, and sync-state tables
- deploy the API service and connect it to PostgreSQL and Redis

Done when:

- `CREATE EXTENSION vector` is enabled
- NLP tables exist with the expected indexes
- the API can connect to PostgreSQL and run a simple vector query

### Step 2: Build the Data Pipeline

What this step covers:

- `fetcher.py`: download bill full text from GovInfo
- `chunker.py`: split bill text into useful semantic chunks
- `embedder.py`: batch embeddings through OpenAI
- `loader.py`: bulk insert chunk rows and embedding rows into PostgreSQL
- `tracker.py`: maintain bill-level sync state for incremental updates

Done when:

- the NLP tables contain a representative sample corpus
- a manual similarity query over stored embeddings returns sensible chunk matches

### Step 3: Build Hybrid Retrieval

What this step covers:

- query normalization and filter extraction
- vector search in `pgvector`
- keyword search in PostgreSQL `tsvector`
- Reciprocal Rank Fusion
- cross-encoder reranking
- metadata hydration

Done when:

- a Python script can take a natural-language query and return ranked bills plus supporting chunks

### Step 4: API and Answer Generation

What this step covers:

- FastAPI endpoint for `/api/nlp/search`
- streaming source results and generated answer text
- Redis caches for embeddings, retrieval results, reranked candidates, and final responses
- prompt construction that cites bill IDs and retrieved passages only

Done when:

- a local request returns source bills first and then a grounded streamed answer

### Step 5: Incremental Sync

What this step covers:

- nightly or scheduled sync against newly added or updated bills
- selective re-fetch, re-chunk, and re-embed for changed source documents
- bookkeeping to avoid reprocessing unchanged bills

Done when:

- a small incremental sync run updates PostgreSQL embeddings without rebuilding the corpus

## Memory and Operations Notes

Moving from Qdrant to `pgvector` simplifies the deployment footprint:

- no separate vector database container
- no Qdrant persistence volume
- fewer moving parts in Kubernetes

But it shifts more load onto PostgreSQL, so production tuning matters:

- `shared_buffers`
- `work_mem`
- autovacuum behavior
- index build strategy
- ANN index choice and maintenance

For a single-node deployment, this tradeoff is acceptable because operational simplicity is more valuable than squeezing out a separate specialized vector tier too early.

## Open Questions

1. Should embeddings live in the main `csearch` database or in a dedicated `csearch_nlp` database on the same PostgreSQL instance?
2. Which ANN index strategy should we use first for `pgvector`: `hnsw` or `ivfflat`?
3. Do we keep one embeddings table for all chunk types, or partition by corpus and year for easier maintenance?
4. Should nightly sync re-embed the full bill on any detected change, or try section-level invalidation?
5. How much of the proof-of-concept code in `project-tarp/` should be promoted directly versus rewritten around PostgreSQL bulk loading?

For deployment details, schema sketches, and concrete SQL/Kubernetes steps, see `IMPLEMENTATION.md`.
