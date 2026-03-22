# CSearch NLP — Natural Language Bill Search

A standalone RAG service for semantic search over U.S. Congressional legislation, designed to run entirely locally using an RTX 3090, Qdrant, and Ollama.

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
    │ PostgreSQL │    │   Qdrant     │   │Ollama (seed) │
    │ (existing, │    │ (on-disk     │   │OpenAI (live) │
    │ read-only) │    │ HNSW mode)   │   │              │
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

### 2. Hybrid Inference Pipeline (Ollama + OpenAI)
To balance massive ingestion volumes with production latency and quality requirements, we split inference between local and cloud providers:
- **Vector DB Seeding (Ollama)**: Because we use hardware equipped with high VRAM (RTX 3090, 24GB VRAM), we utilize dense local embedders (`nomic-embed-text` or `mxbai-embed-large`) via Ollama. This eliminates usage limits and external API costs when bulk-loading and embedding millions of document chunks into the Qdrant DB.
- **Production API Hits (OpenAI)**: When a user performs a search, the live query embedding and final synthesized LLM answer generation are routed through the OpenAI API. This ensures low-latency, highly accurate responses for the actual user-facing application without bogging down the local GPU.

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

**What this covers:**
- Finalize the embedding model. The current README references Ollama for batch seeding and OpenAI for live queries, but `NLP.md` specifies `text-embedding-3-small` (OpenAI, 1536 dims) for everything in production and Ollama only for local dev. These need to agree, because the Qdrant collection's vector dimensions are locked at creation time. If we go with Ollama's `nomic-embed-text` (768 dims) for dev and OpenAI (1536 dims) for prod, the collections are incompatible — we'd need to re-embed when migrating.
- Finalize the LLM provider. `NLP.md` specifies Claude Sonnet for answer generation; the README references OpenAI generically; `IMPLEMENTATION.md` configmap uses `qwen2.5:7b` via Ollama. Pick one for dev and one for prod, or settle on a single provider.
- Finalize the storage backend. `IMPLEMENTATION.md` uses `hostPath` PVs on local SSD. `NLP.md` uses NFS-backed PVs (`nfs-client` storageClassName on `10.0.0.3`). For `mars` dev with an attached SSD, `hostPath` is simpler and faster. For `netcup` prod, NFS or a CSI driver may be required. Decide now so the K8s manifests don't need rewriting later.
- Create the `csearch-nlp` namespace, PVs/PVCs, and secrets on `mars`. Wire up ArgoCD so that from this point forward, all K8s resource changes flow through git commits.
- Create the read-only PostgreSQL role (`csearch_readonly`) with `SELECT`-only grants.
- Deploy Qdrant (StatefulSet + Service), verify it's healthy, and create the `bill_chunks` collection with the agreed-upon vector dimensions, INT8 quantization, and on-disk HNSW config.

**Done when:** Qdrant is running on `mars`, the collection exists with correct dimensions and indexes (`billid`, `congress`, `billtype`, `year`, `chunk_type`), ArgoCD is syncing the `k8s/` directory, and all secrets are applied.

---

### Step 2 — Build the Data Ingestion Pipeline

This is the most time-intensive step. The goal is to go from "empty Qdrant collection" to "~8M searchable vectors" covering 50 years of Congressional legislation.

**What this covers:**
- **Fetcher** (`pipeline/fetcher.py`): Download full bill XML from GovInfo's bulk data endpoint (`https://www.govinfo.gov/bulkdata/BILLS/{congress}/{billtype}/`). Rate-limit at ~10 req/s with exponential backoff. Cache all XML locally (NFS or dev machine disk) so re-runs don't re-download. Only fetch the latest version of each bill (enrolled > engrossed > reported > introduced).
- **Chunker** (`pipeline/chunker.py`): XML-aware section splitting. Preserve `<section>` boundaries (don't split mid-section if < 512 tokens). Split long sections at `<paragraph>` or `<subsection>` elements with 64-token overlap. Prepend bill context to every chunk (`"[H.R. 1234, 118th Congress] Section 3: Definitions — "`). Filter out boilerplate sections (effective date, severability, authorization of appropriations, short title). Extract definitions sections as their own dedicated chunks.
- **Chunk types to ingest**: Titles (~500K), Summaries (~1M), Full-text substantive sections (~5M), Definitions (~500K), Actions timeline (~1.5M), Sponsor blocks (~500K). Estimated total: ~8M chunks.
- **Batcher** (`pipeline/batcher.py`): Batch embed chunks via Ollama (dev) or OpenAI API (prod). On the dev machine with RTX 3090, use `nomic-embed-text` via Ollama to avoid API costs entirely. 4 parallel workers, batch size 100.
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
- **Generator** (`rag/generator.py`): Stream the LLM response. Use Claude Sonnet (prod) or Ollama `qwen2.5:7b` (dev). Per-query context is ~5–10K tokens (10 bills × 1–3 matched chunks + metadata).
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
| Qdrant | 1.5 GB | 2 GB | INT8 quantized vectors in RAM, HNSW + payloads on disk |
| csearch-nlp API | 1 GB | 2 GB | Includes cross-encoder model (~500 MB) |
| Redis (existing) | ~256 MB | ~512 MB | NLP adds ~50 MB for caches |
| OS + K8s overhead | ~512 MB | — | kubelet, kernel, etc. |
| **Total (steady state)** | **~4.3 GB** | **~7 GB** | Fits in 8 GB with headroom |

The midnight goscraper CronJob and 00:30 NLP sync CronJob overlap briefly. Both are short-lived and bursty. The NLP sync's conservative limits (512Mi/1Gi) keep peak overlap safe.

---

## Open Questions

These need to be resolved before or during implementation. Numbered for easy reference in future discussions.

**Q1 — Embedding model consistency across dev and prod.**
If dev uses Ollama `nomic-embed-text` (768 dims) and prod uses OpenAI `text-embedding-3-small` (1536 dims), the Qdrant collections are structurally incompatible — you'd have to re-embed the entire corpus when migrating to prod. Options: (a) use OpenAI for both dev and prod (accept the ~$32 batch cost), (b) use Ollama for both and accept slightly lower embedding quality in prod, or (c) treat the dev collection as throwaway and plan a full re-embed for prod. Which approach?

**Q2 — Ollama networking in K8s.**
Ollama runs on the local machine with the RTX 3090. Is it a native host process or containerized? The K8s pods (API and sync CronJob) need a reachable endpoint — either `http://host.docker.internal:11434`, a K8s Service pointing to a host port, or a NodePort. What's the current Ollama setup?

**Q3 — LLM provider for generation.**
Three options appear across the docs: Claude Sonnet (NLP.md), OpenAI (README), and Ollama `qwen2.5:7b` (IMPLEMENTATION.md configmap). For dev, Ollama is free but lower quality. For prod, Claude or OpenAI gives much better answer synthesis. Is the plan to use Ollama for dev iteration and Claude Sonnet for prod? Should the code abstract over both via a common interface?

**Q4 — Git hosting and ArgoCD.**
ArgoCD needs a reachable git remote to sync from. Is the `csearch-nlp` repo going on GitHub/GitLab, or is there a self-hosted Gitea/Forgejo instance on the cluster? If it's not yet on a remote, ArgoCD can't auto-sync — deployment would be manual `kubectl apply` until the repo is pushed upstream.

**Q5 — Storage backend: hostPath vs. NFS.**
`IMPLEMENTATION.md` uses `hostPath` PVs (fast, local SSD). `NLP.md` uses NFS PVs (`nfs-client` on `10.0.0.3`). For Qdrant with on-disk HNSW, SSD latency matters — NFS adds a network hop that could push cold-cache queries from ~150ms to 300ms+. Is the `mars` dev node equipped with local SSD? Should we use `hostPath` for `mars` and defer the NFS decision to `netcup` prod?

**Q6 — Production migration strategy.**
The prod environment (`netcup`) is a separate remote VPS without a GPU. When it's time to deploy there, the Qdrant DB either needs to be (a) re-embedded from scratch using API-based embeddings, (b) snapshot-synced from `mars`, or (c) served from `mars` and proxied. How much thought should go into this now vs. after the dev pipeline stabilizes?

**Q7 — Vote data.**
The README and Phase 1 CLI commands reference `--include-votes` and a `vote_chunks` collection, but `NLP.md` only defines a `bill_chunks` collection. Are votes a separate collection, a chunk type within `bill_chunks`, or deferred to a later phase?

**Q8 — API authentication.**
For `mars` dev, no auth is needed. But once this is exposed via the frontend in production, should the `/api/nlp/search` endpoint require authentication? Rate limiting? The LLM generation step is the most expensive per-request — an unauthenticated endpoint could run up costs quickly.

**Q9 — Cross-encoder model size on VPS.**
The cross-encoder (`ms-marco-MiniLM-L-12-v2`, ~130MB on disk, ~500MB under load) runs on CPU within the API pod. With the API pod limited to 2 GB, and the model taking ~500 MB, that leaves ~1.5 GB for the FastAPI process, async request handling, and Python overhead. Is this sufficient under concurrent load, or should we consider a lighter model or a separate sidecar?
