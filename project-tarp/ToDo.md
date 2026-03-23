# Project TARP — Roadmap & ToDo

Last updated: 2026-03-23

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| `fetcher.py` | ✅ Complete | 7,960 bills discovered, 7,789 XML files downloaded |
| `chunker.py` | ✅ Complete | Full Congress 110 run completed; 439,890 chunks across 21 shards with section dedup enabled |
| `embedder.py` | ✅ Complete | Full embedding run completed to sharded output under `data/embedded_chunks/` |
| `upserter.py` | 🟡 In progress | Implemented; currently upserting embedded shards into Qdrant |
| `query.py` | ❌ Not started | Needs to be written |
| Qdrant | 🟡 Running | Deployed on `mars` in namespace `csearch-nlp` via LoadBalancer at `192.168.1.156:6333` |
| OpenAI API key | ✅ Set | Used for the completed embedding run |

---

## Phase 1: Finish Downloading (In Progress — ~80%)

- [x] Build and test `fetcher.py`
- [x] Download HR + S bills for Congress 110
- [ ] **Verify the fetcher has completed** — check the final count and confirm no errors
  ```bash
  # Check how many bills were discovered vs downloaded
  python fetcher.py --congresses 110 --dry-run
  # Look at the fetch manifest for error/not_found counts
  python3 -c "import json; d=json.load(open('data/bills_110/fetch_manifest.json')); print({r['status'] for r in d}); print({s: sum(1 for r in d if r['status']==s) for s in set(r['status'] for r in d)})"
  ```
- [ ] **Re-run the fetcher** for any bills that errored out (network timeouts, etc.)
  ```bash
  python fetcher.py --congresses 110
  # It will auto-skip already-downloaded bills (checks for .meta.json)
  ```

---

## Phase 2: Chunking ✅ COMPLETE

The chunker has been run on the full Congress 110 dataset and now writes deduplicated JSONL shards plus a manifest.

- [x] **Install tiktoken** if not already present (the chunker falls back to word-count estimation without it, but real token counts are critical for accurate cost estimates)
  ```bash
  pip install tiktoken
  ```
- [x] **Run the chunker on the full dataset**
  ```bash
  cd project-tarp
  python chunker.py --congresses 110
  ```
  Current chosen run used `--max-chunks-per-bill 300` and writes shard output under `data/processed_chunks/110/`.
- [x] **Inspect the output** — verify chunk quality before spending money on embeddings
  ```bash
  cat data/processed_chunks/110/manifest.json
  ```
  **Observed output:** 439,890 chunks across 21 shards, 10,493 canonical bills with chunks, 11,571 duplicate sections removed, ~55.3M tokens for embedding.
- [x] **Spot-check 5–10 chunks manually** to verify:
  - Context prefix is correct (e.g., `[H.R. 1424, 110th Congress] Section 101: ...`)
  - Boilerplate sections (short title, effective date) are being filtered out
  - Long sections are split cleanly at subsection boundaries
  - No garbage text or XML tags leaking through
- [x] **Decide: is the chunking quality good enough?** Current answer: yes for the POC. Final chosen knob changes were:
  - `--max-tokens` (default 512) — smaller = more chunks but tighter semantic focus
  - `--overlap` (default 64) — more overlap = better cross-boundary recall but higher cost
  - `--max-chunks-per-bill 300` — higher cap after testing showed 200 was too aggressive for monster bills

---

## Phase 3: Embedding ✅ COMPLETE

Embedding is complete and now writes mirrored JSONL shards under `data/embedded_chunks/` instead of one giant checkpoint file.

- [x] **Set the OpenAI API key**
  ```bash
  export OPENAI_API_KEY='sk-...'
  ```
- [x] **Do a dry run first** to confirm the cost estimate
  ```bash
  cd project-tarp
  python embedder.py --dry-run
  ```
- [x] **Run the embedder**
  ```bash
  python embedder.py
  ```
  This will:
  - Read `data/processed_chunks/`
  - Send batches to OpenAI `text-embedding-3-small`
  - Save mirrored results to `data/embedded_chunks/`
  - Resume shard-locally if interrupted
- [x] **Verify the embedded output**
  ```bash
  find data/embedded_chunks -maxdepth 2 -name 'shard-*.jsonl' | head
  ```
  Confirmed: embedded shards exist, vectors are 1536-dim, and output size is ~13G.

---

## Phase 4: Qdrant Setup & Upsert 🟡 IN PROGRESS

- [x] **Deploy Qdrant on `mars` Kubernetes**
  - Namespace: `csearch-nlp`
  - Service type: `LoadBalancer`
  - External REST endpoint: `http://192.168.1.156:6333`
  - External gRPC endpoint: `192.168.1.156:6334`
- [x] **Write `upserter.py`**
  - Reads `data/embedded_chunks/`
  - Uses deterministic UUID point IDs
  - Defaults to Qdrant host `192.168.1.156:6333`
  - Targets collection `bill_chunks`
  - Keeps payloads lean by default
- [ ] **Run the upserter**
  ```bash
  cd project-tarp
  python upserter.py
  ```
- [ ] **Verify collection health after ingestion completes**
  ```bash
  python3 -c "
  from qdrant_client import QdrantClient
  c = QdrantClient('192.168.1.156', port=6333)
  info = c.get_collection('bill_chunks')
  print(f'Points: {info.points_count}')
  print(f'Vectors: {info.vectors_count}')
  print(f'Status: {info.status}')
  "
  ```

---

## Phase 5: Query Engine & Answer Generation

- [ ] **Write `query.py`** — interactive CLI that:
  1. Accepts a natural language query from stdin
  2. Embeds the query via OpenAI `text-embedding-3-small` (single API call, ~$0.000002)
  3. Searches Qdrant `bills_2008_test` for the top 5 closest vectors
  4. Displays the matched chunks with scores, bill IDs, section headers, and text snippets
  5. Passes the top 5 chunks as context to `gpt-5.4-nano` with a prompt like:
     > "Based on the following excerpts from congressional bills, answer the user's question. Cite specific bill numbers. Do not fabricate information."
  6. Prints the generated answer to the terminal
- [ ] **Run test queries** to validate end-to-end:
  ```bash
  cd project-tarp
  python query.py
  # Try these queries:
  # "What did Congress do about the financial crisis in 2008?"
  # "bills about bank bailouts"
  # "legislation regulating subprime mortgages"
  # "environmental protection bills from the 110th Congress"
  # "bills related to veterans healthcare"
  ```
- [ ] **Evaluate quality** — are the retrieved chunks actually relevant? Is the generated answer citing real bill numbers from the results?

---

## Phase 6: Polish & Document

- [ ] **Add a `requirements.txt`** to project-tarp
  ```
  openai
  tiktoken
  qdrant-client
  ```
- [x] **Update `PLAN.md`** with actual results (chunk counts, cost, quality observations)
- [ ] **Write findings** — what worked, what didn't, what to change before scaling to all 50 years
- [ ] **Commit and push** the final project-tarp code (excluding data files)

---

## Decision Points (Pause & Think)

These are moments where you should stop and evaluate before continuing:

1. **After Phase 2 (chunking):** Are the chunks clean? Is the cost estimate reasonable? Do not proceed to Phase 3 until you're satisfied with chunk quality.
2. **After Phase 3 (embedding):** Did the API calls complete without errors? Is the checkpoint file intact? Verify vector dimensions before proceeding.
3. **After Phase 5 (first queries):** Do the search results make semantic sense? Is the LLM answer grounded in the retrieved chunks, or is it hallucinating? This determines whether the pipeline is viable at scale.

---

## Future (Post-POC, Back to `csearch-nlp` Main)

These items are explicitly out of scope for Project TARP but are recorded here so nothing is lost:

- [ ] Scale to all 50 years of Congress (93rd–118th)
- [ ] Evaluate OpenAI Batch API for 50% cost reduction on the full corpus
- [ ] Add `vote_chunks` collection for roll call vote data
- [ ] Hybrid retrieval (Qdrant vectors + PostgreSQL keyword search + RRF fusion)
- [ ] Cross-encoder reranking for top-k refinement
- [ ] Redis caching layer
- [ ] FastAPI service with SSE streaming
- [ ] K8s deployment on `mars` with ArgoCD
- [ ] Nightly sync CronJob for new bills
- [ ] Frontend integration with CSearch Nuxt UI
