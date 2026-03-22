# CSearch NLP — Natural Language Bill Search

A standalone RAG service for semantic search over U.S. Congressional legislation. Embeddings are generated via OpenAI's `text-embedding-3-small` and stored in Qdrant. Answer generation uses `gpt-5.4-nano`. The service runs on a local K8s cluster (`mars`) with hostPath SSD storage.

## Problem Statement

CSearch currently uses PostgreSQL full-text search (`tsvector` with GIN indexes) over `shorttitle` and `summary->>'Text'`. This handles keyword queries effectively but fails on subjective or semantic queries, such as:
- "bills that would make it harder for companies to pollute rivers"
- "legislation protecting gig workers from being classified as independent contractors"
- "what has Congress done about prescription drug prices since 2020"

This RAG pipeline supplements the system by combining vector similarity search with LLM-powered answer generation. It maps natural language questions against the full text of 50+ years of Congressional legislation (Bills + Votes).

This is a **standalone project** with read-only access to the central CSearch PostgreSQL database, its own Qdrant vector database, and an API surface isolated from the core backend.

---

## Architecture Conceptual Overview

```text
                    ┌─────────────────────────────┐
                    │    CSearch Frontend (Nuxt)  │
                    └──────────────┬──────────────┘
                                   │
                          POST /api/nlp/search
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              csearch-nlp service (FastAPI)                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                     RAG Orchestrator                       │  │
│  │                                                            │  │
│  │  1. Query classifier — extract filters, classify intent    │  │
│  │  2. Embed query via OpenAI API                             │  │
│  │  3. Retrieve from Qdrant (filtered ANN, on-disk HNSW)      │  │
│  │  4. Keyword search against PostgreSQL tsvector (parallel)  │  │
│  │  5. Reciprocal Rank Fusion to merge results                │  │
│  │  6. Cross-encoder re-rank (ms-marco-MiniLM-L-12-v2)        │  │
│  │  7. Hydrate bill metadata from PostgreSQL                  │  │
│  │  8. Stream LLM response via OpenAI API                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Redis (shared, nlp: prefix): embedding + result + LLM cache     │
└──────────┬──────────────────┬──────────────────┬─────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌────────────┐    ┌──────────────┐   ┌──────────────┐
    │ PostgreSQL │    │   Qdrant     │   │   OpenAI     │
    │ (existing, │    │ (on-disk     │   │  embed +     │
    │ read-only) │    │ HNSW mode)   │   │  generate    │
    └────────────┘    └──────────────┘   └──────────────┘
           │
     read-only
           ▼
    ┌────────────┐
    │  GovInfo   │
    │ (full bill │
    │  text XML) │
    └────────────┘
```

## The Moving Pieces

### 1. Extraction and Filtering
Embedding 50 years of raw Congressional `.xml` creates an unmanageable ~55M context chunks. To make this locally hostable without blowing out RAM/Disk:
- We selectively ingest high-value zones: **Titles**, **Summaries**, **Active Text sections**, **Definitions**, and **Sponsor/Cosponsor Data**.
- We skip: Boilerplate (effective dates, authorization fluff) and obsolete drafts (Introduced versions if Enrolled exists).
- Incorporating JSON Vote records allows mapping semantic clusters of historical "Yea/Nay" dynamics.
- *Outcome:* The graph is reduced 7x to roughly **~10M chunks**.

### 2. OpenAI Inference Pipeline
All inference is routed through OpenAI for consistency between dev and prod environments:
- **Embeddings** (`text-embedding-3-small`, 1536 dims): Used for both batch ingestion and live query embedding. This ensures structural compatibility between the dev Qdrant collection and any future production deployment — no re-embedding required when migrating. Estimated batch cost for the full 50-year corpus is ~$32; single-year test runs (e.g., 110th Congress / 2008) cost ~$1.
- **Answer Generation** (`gpt-5.4-nano`): The cheapest and fastest model in the GPT-5.4 family ($0.20/1M input tokens). Used to synthesize a final user-facing answer from the top retrieved bill chunks. Quality is excellent for structured summarization tasks.

### 3. Vector Datastore (Qdrant)
We utilize Qdrant in **Memory-Mapped (HostPath) Mode** to maintain tight RAM footprints.
- Raw payloads and the HNSW graph remain stored on SSD (`/mnt/data/qdrant`).
- **INT8 Quantization** loads 1-byte simplified context footprints into memory, keeping the entire search engine alive. 

### 4. RAG Orchestration (Python/FastAPI)
Search blends two methodologies together:
1. Keyword Hits (From PostgreSQL `tsvector`).
2. Vector Proximity (From Qdrant).
3. **Reciprocal Rank Fusion (RRF)** combines these lists mathematically for the most robust resultset without depending purely on similarity space.
4. An intermediary **Cross-Encoder Model** double-checks the top 40 outputs line-by-line vs the query to surface the absolute best 10 Bills.
5. Context is sent sequentially back to the Frontend UI.

---

For exact command instructions, K8s manifests, and the full backend schema layouts, refer to `IMPLEMENTATION.md`.

---

## Implementation Plan

This plan is organized into big conceptual steps. Each step is a meaningful milestone — the project should be testable (or at least verifiable) at the end of each one.

### Step 1 — Resolve Infrastructure Foundations

Before writing any application code, nail down the decisions that everything else depends on.

**Decisions resolved:**
- **Embedding model:** OpenAI `text-embedding-3-small` (1536 dims) for both dev and prod.
- **LLM provider:** OpenAI `gpt-5.4-nano` for answer generation.
- **Storage backend:** `hostPath` PVs on local SSD for `mars`. Likely hostPath for prod too.
- **Git hosting:** GitHub repo for ArgoCD sync.
- **Vote data:** Separate `vote_chunks` Qdrant collection alongside `bill_chunks`.

**What this covers:**
- Create the `csearch-nlp` namespace, PVs/PVCs, and secrets on `mars`. Wire up ArgoCD to sync from the GitHub repo so that all K8s resource changes flow through git commits.
- Create the read-only PostgreSQL role (`csearch_readonly`) with `SELECT`-only grants.
- Deploy Qdrant (StatefulSet + Service), verify it's healthy, and create the `bill_chunks` collection with 1536-dim vectors, INT8 quantization, and on-disk HNSW config. Create a separate `vote_chunks` collection with the same config.

**Done when:** Qdrant is running on `mars`, both collections exist with correct dimensions and payload indexes (`billid`, `congress`, `billtype`, `year`, `chunk_type`), ArgoCD is syncing the `k8s/` directory from GitHub, and all secrets are applied.

---

### Step 2 — Build the Data Ingestion Pipeline

This is the most time-intensive step. The goal is to go from "empty Qdrant collection" to "~8M searchable vectors" covering 50 years of Congressional legislation.

**What this covers:**
- **Fetcher** (`pipeline/fetcher.py`): Download full bill XML from GovInfo's bulk data endpoint (`https://www.govinfo.gov/bulkdata/BILLS/{congress}/{billtype}/`). Rate-limit at ~10 req/s with exponential backoff. Cache all XML locally (NFS or dev machine disk) so re-runs don't re-download. Only fetch the latest version of each bill (enrolled > engrossed > reported > introduced).
- **Chunker** (`pipeline/chunker.py`): XML-aware section splitting. Preserve `<section>` boundaries (don't split mid-section if < 512 tokens). Split long sections at `<paragraph>` or `<subsection>` elements with 64-token overlap. Prepend bill context to every chunk (`"[H.R. 1234, 118th Congress] Section 3: Definitions — "`). Filter out boilerplate sections (effective date, severability, authorization of appropriations, short title). Extract definitions sections as their own dedicated chunks.
- **Chunk types to ingest**: Titles (~500K), Summaries (~1M), Full-text substantive sections (~5M), Definitions (~500K), Actions timeline (~1.5M), Sponsor blocks (~500K). Estimated total: ~8M chunks.
- **Batcher** (`pipeline/batcher.py`): Batch embed chunks via OpenAI `text-embedding-3-small`. Use batches of 500–1000 strings per API call. Checkpoint embedded results to disk so failed runs can resume without re-paying for already-embedded chunks.
- **Upserter** (`pipeline/upserter.py`): Stream embedded vectors to Qdrant over a `kubectl port-forward` from the dev machine. The dev machine does the heavy lifting; only final vectors cross the wire.
- **Tracker** (`pipeline/tracker.py`): Track which bills have been embedded (sync state) so incremental runs only process new/updated bills.

**Done when:** `qdrant.get_collection('bill_chunks')` reports ~8M points, and a quick manual Qdrant search returns sensible results for a test query vector.

---

### Step 3 — Build the RAG Retrieval Pipeline

With vectors in Qdrant, build the query-time retrieval path — everything between "user types a question" and "we have a ranked list of 10 relevant bills with metadata."

**What this covers:**
- **Query classifier** (`rag/query_classifier.py`): Extract structured filters from natural language (congress number, bill type, year range, sponsor name). Can be a lightweight LLM call (Haiku-class) or regex heuristics for common patterns. Clean the query text of extracted filter terms before embedding.
- **Embedder** (`rag/embedder.py`): Embed the cleaned query using the same model as ingestion. For production, this must match the model used at ingest time — mixing embedding models produces garbage results.
- **Retriever** (`rag/retriever.py`): Run two searches in parallel — (a) Qdrant filtered ANN search (top_k=40, with payload filters from the classifier) and (b) PostgreSQL `tsvector` keyword search against `shorttitle` and `summary->>'Text'` (top 20 bill IDs).
- **Fusion** (`rag/fusion.py`): Merge the two result lists using Reciprocal Rank Fusion (k=60). RRF is rank-based, not score-based, so it naturally handles the different scales of cosine similarity vs. `ts_rank`.
- **Reranker** (`rag/reranker.py`): Cross-encoder re-rank the top 40 fused results using `cross-encoder/ms-marco-MiniLM-L-12-v2` (~130MB, CPU-only, ~80ms for 40 pairs). This is not optional — at 8M chunks, bi-encoder retrieval returns near-misses that need joint (query, chunk) scoring to surface the best 10.
- **Hydrator** (`rag/hydrator.py`): Fetch full bill metadata from PostgreSQL for the top 10 bills (sponsors, status, introduced date, summary text, etc.).

**Done when:** A Python script can take a natural language query string and return 10 ranked bills with metadata, without any API server or LLM generation involved. Test with queries like "bills banning stock trading by members of Congress" and "prescription drug pricing since 2020."

---

### Step 4 — Build the API and Generation Layer

Wrap the retrieval pipeline in a FastAPI service and add LLM-powered answer generation with streaming.

**What this covers:**
- **FastAPI app** (`api/server.py`, `api/routes.py`, `api/models.py`): `POST /api/nlp/search` accepting a query string and optional filters. `GET /health` for liveness/readiness probes.
- **Prompt builder** (`rag/prompt_builder.py`): Construct the LLM prompt from the 10 ranked bills. Template includes: bill number, title, congress, introduced date, status, sponsors, matched chunk text with section titles, and summary. Rules instruct the LLM to cite specific bill numbers, quote statutory language, compare approaches, and never fabricate.
- **Generator** (`rag/generator.py`): Stream the LLM response via OpenAI `gpt-5.4-nano`. Per-query context is ~5–10K tokens (10 bills × 1–3 matched chunks + metadata). Cost per query is negligible (~$0.001).
- **SSE streaming format**: Three event types — `sources` (bill list with scores, sent immediately after retrieval), `token` (incremental LLM text), `done` (usage stats). This lets the frontend show source bills before the answer finishes generating.
- **Caching** (`cache/redis_cache.py`): Layer caches on the shared CSearch Redis with `nlp:` key prefix. Cache query embeddings (24h TTL), Qdrant+RRF results (1h), re-ranked results (1h), LLM responses (1h), and bill metadata (6h). Critical for on-disk Qdrant — repeated queries hit Redis (~1ms) instead of disk HNSW (~150ms).
- **Dockerize**: Build the image with the cross-encoder model baked in (downloaded at build time). Push to the private registry at `10.0.0.3:30252`.
- **Deploy**: Update the K8s deployment image tag, commit, let ArgoCD sync. Verify via port-forward.

**Done when:** `curl -X POST http://localhost:8000/api/nlp/search -d '{"query": "..."}'` returns a streaming SSE response with source bills and a coherent LLM-generated answer.

---

### Step 5 — Nightly Sync and Incremental Updates

The initial batch covers historical data. New bills are introduced daily, and the vector DB needs to stay current.

**What this covers:**
- The nightly CronJob (`k8s/nlp-sync-cronjob.yaml`) runs at 00:30, after the existing goscraper finishes at midnight. It calls `python -m csearch_nlp.pipeline sync`, which queries PostgreSQL for bills added/updated since the last sync, fetches their XML, chunks, embeds, and upserts to Qdrant.
- Daily volume is small (~100 new bills/day = ~10K chunks = ~2 minutes of work). Lightweight enough to run on the VPS itself.
- Resource limits are conservative (512Mi request / 1Gi limit) to avoid memory pressure from overlapping with the goscraper CronJob.
- Test by manually creating a job from the CronJob and tailing logs.

**Done when:** A manually triggered sync job completes successfully, and newly added bills appear in search results.

---

### Step 6 — Frontend Integration

Connect the NLP search API to the CSearch user interface.

**What this covers:**
- The API is reachable within the cluster at `csearch-nlp-api.csearch-nlp.svc.cluster.local:8000`. Two integration approaches: (a) have the Nuxt frontend call it directly (if the frontend has a server-side proxy layer), or (b) have the existing Fastify backend proxy requests to it.
- Frontend needs to handle the SSE streaming format: display source bills immediately, then stream the LLM answer text as tokens arrive.
- Consider a UI mode that lets users toggle between the existing keyword search and the new NLP search, so they can fall back if results are poor.

**Done when:** A user can type a natural language question in the CSearch UI, see relevant bills populate, and read a streamed LLM-generated answer.

---

### Step 7 — Evaluation and Quality Tuning

Measure retrieval quality and iterate on the pipeline.

**What this covers:**
- Build an eval dataset (`tests/eval/eval_queries.json`): 100+ natural language queries with expected bill IDs. Include diverse query types — topical ("climate change"), sponsor-based ("bills by Senator Warren"), temporal ("since 2020"), and subjective ("bills that weaken environmental protections").
- Run `recall@10` and MRR metrics against the dataset. Baseline these before tuning anything.
- Tuning levers: chunk overlap size, RRF k parameter, cross-encoder top_k, score threshold, which chunk types to include/exclude, embedding model choice.
- LLM output quality: spot-check for hallucinated bill numbers (validate every cited ID against the retrieved set in post-processing), completeness of answers, and citation accuracy.

**Done when:** recall@10 and MRR are baselined, and a repeatable eval pipeline (`run_eval.py`) exists to measure the impact of any future changes.

---

## Memory Budget (VPS Steady State)

All services share a single 4 CPU / 8 GB RAM VPS node. Approximate memory allocations at steady state:

| Process | Requests | Limits | Notes |
|---|---|---|---|
| PostgreSQL (existing) | ~1 GB | ~2 GB | Unchanged |
| Qdrant | 1.5 GB | 2 GB | INT8 quantized vectors in RAM, HNSW + payloads on disk (hostPath SSD) |
| csearch-nlp API | 1 GB | 2 GB | Lighter footprint if cross-encoder is replaced by OpenAI reranking |
| Redis (existing) | ~256 MB | ~512 MB | NLP adds ~50 MB for caches |
| OS + K8s overhead | ~512 MB | — | kubelet, kernel, etc. |
| **Total (steady state)** | **~4.3 GB** | **~7 GB** | Fits in 8 GB with headroom |

The midnight goscraper CronJob and 00:30 NLP sync CronJob overlap briefly. Both are short-lived and bursty. The NLP sync's conservative limits (512Mi/1Gi) keep peak overlap safe.

---

## Open Questions

Previous questions (Q1–Q9) have been resolved and archived in [`docs/ANSWERED.md`](docs/ANSWERED.md). The following are new implementation-level questions that need to be addressed during development.

**Q10 — GovInfo XML schema variations across Congress numbers.**
The XML structure of bills has evolved over 50 years. Early Congresses (93rd–100th) may use different tag names, nesting structures, or entirely different document formats than modern ones (110th+). The chunker built for Project TARP targets `<section>`, `<paragraph>`, and `<subsection>` tags from 110th Congress XML. How much structural variation exists, and should the chunker have Congress-era-specific parsing paths, or can a single parser handle all eras with graceful fallbacks?

**Q11 — OpenAI Batch API vs. standard API for initial embedding.**
OpenAI's Batch API costs 50% less ($0.01/1M tokens vs. $0.02/1M) but processes within a 24-hour window rather than real-time. For the initial bulk embedding of ~10M chunks, the Batch API could cut the full corpus cost from ~$32 to ~$16. Should Project TARP use the Batch API to validate the workflow, or stick with the standard API for faster iteration?

**Q12 — Vote chunk schema design.**
Votes are confirmed as a separate `vote_chunks` collection. But what text actually gets embedded for a vote? Options include: (a) the vote question/description text only, (b) the question text plus a summary of the outcome ("Passed 220-215"), (c) the question text plus the full member roll call serialized as text. Option (c) would be enormous. What level of vote detail is useful for semantic search?

**Q13 — Chunk deduplication strategy.**
Bills go through multiple versions (Introduced → Reported → Engrossed → Enrolled). The plan is to embed only the latest version, but "latest" isn't always "best" — an Introduced version of a bill that died in committee may be the only version that exists. Additionally, companion bills (House + Senate versions of the same legislation) have near-identical text. Should we deduplicate across companion bills, or let the vector DB naturally cluster them and rely on the reranker to collapse duplicates?

**Q14 — OpenAI API key management for the pipeline.**
Project TARP runs locally on the dev machine, not in K8s. The OpenAI API key needs to be available to the Python scripts. Should it be loaded from a `.env` file (gitignored), an environment variable, or pulled from the existing K8s secrets store? For the eventual K8s deployment, the key will live in `csearch-nlp-secrets`, but the local pipeline needs its own access pattern.

**Q15 — Qdrant collection indexing strategy.**
The plan calls for payload indexes on `billid`, `congress`, `billtype`, `year`, and `chunk_type`. Qdrant supports keyword, integer, and float payload indexes. Should `congress` and `year` be integer indexes (enabling range queries like "Congress 110–118") or keyword indexes (simpler but no range support)? Should we add a `sponsor` keyword index to support sponsor-filtered searches?

**Q16 — Chunk overlap token count tuning.**
The current plan uses 64-token overlap between split chunks. This is a guess. Too little overlap and the model misses context that spans chunk boundaries; too much overlap inflates the total chunk count (and therefore embedding cost). Should Project TARP test multiple overlap values (0, 32, 64, 128) on a small sample and compare retrieval quality before committing to the full batch?

**Q17 — Error recovery for partial embedding runs.**
If the OpenAI embedding batch fails midway through (network error, rate limit, API outage), the pipeline needs to resume from where it left off without re-embedding already-processed chunks. The current plan checkpoints to `embedded_chunks.json`, but for 10M chunks this file would be enormous (~50GB+). Should the tracker use a lightweight SQLite database or a simple offset log instead of a monolithic JSON file?

**Q18 — Nightly sync: OpenAI cost for daily incremental updates.**
The nightly sync job embeds ~100 new bills/day (~10K chunks). At `text-embedding-3-small` pricing, this is ~$0.005/day or ~$1.80/year — effectively free. But should the sync job have a hard spending cap or alert threshold to catch unexpected spikes (e.g., if a bulk data dump suddenly adds thousands of historical bills)?