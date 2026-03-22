# Project TARP — Roadmap & ToDo

Last updated: 2026-03-22

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| `fetcher.py` | ✅ Complete | 7,960 bills discovered, 7,789 XML files downloaded |
| `chunker.py` | ✅ Code complete | Tested on 20 bills → 432 chunks. Needs full run on all 7,789 bills |
| `embedder.py` | ✅ Code complete | Not yet run (waiting on full chunk set) |
| `upserter.py` | ❌ Not started | Needs to be written |
| `query.py` | ❌ Not started | Needs to be written |
| Qdrant | ❌ Not running | Docker container not yet started |
| OpenAI API key | ❓ Unknown | Needs to be set as `OPENAI_API_KEY` env var |

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

## Phase 2: Chunking (Next Up)

The chunker code is complete and tested on a 20-bill sample (produced 432 chunks). Now it needs to process all ~7,789 downloaded bills.

- [ ] **Install tiktoken** if not already present (the chunker falls back to word-count estimation without it, but real token counts are critical for accurate cost estimates)
  ```bash
  pip install tiktoken
  ```
- [ ] **Run the chunker on the full dataset**
  ```bash
  cd project-tarp
  python chunker.py --congresses 110
  ```
  This will overwrite `data/processed_chunks.json` with all chunks from all 7,789 bills.
- [ ] **Inspect the output** — verify chunk quality before spending money on embeddings
  ```bash
  # Summary stats
  python3 -c "
  import json
  d = json.load(open('data/processed_chunks.json'))
  tokens = sum(c['tokens'] for c in d)
  bills = len(set(c['bill_id'] for c in d))
  print(f'Total chunks: {len(d):,}')
  print(f'Unique bills: {bills:,}')
  print(f'Total tokens: {tokens:,}')
  print(f'Avg tokens/chunk: {tokens//len(d)}')
  print(f'Estimated embedding cost: \${tokens/1_000_000*0.02:.4f}')
  "
  ```
  **Expected output:** ~100K–200K chunks, ~$0.50–$1.00 estimated cost.
- [ ] **Spot-check 5–10 chunks manually** to verify:
  - Context prefix is correct (e.g., `[H.R. 1424, 110th Congress] Section 101: ...`)
  - Boilerplate sections (short title, effective date) are being filtered out
  - Long sections are split cleanly at subsection boundaries
  - No garbage text or XML tags leaking through
  ```bash
  # Print a few random chunks
  python3 -c "
  import json, random
  d = json.load(open('data/processed_chunks.json'))
  for c in random.sample(d, 5):
      print(f\"--- {c['bill_id']} §{c['section_enum']} [{c['tokens']} tok] ---\")
      print(c['text'][:300])
      print()
  "
  ```
- [ ] **Decide: is the chunking quality good enough?** If not, tune these knobs:
  - `--max-tokens` (default 512) — smaller = more chunks but tighter semantic focus
  - `--overlap` (default 64) — more overlap = better cross-boundary recall but higher cost
  - `--max-chunks-per-bill` (default 200) — safety cap for enormous bills

---

## Phase 3: Embedding (OpenAI API Call)

This is the step that costs real money (~$0.50–$1.00 for Congress 110). Do not rush into this until Phase 2 output looks clean.

- [ ] **Set the OpenAI API key**
  ```bash
  export OPENAI_API_KEY='sk-...'
  ```
- [ ] **Do a dry run first** to confirm the cost estimate
  ```bash
  cd project-tarp
  python embedder.py --dry-run
  ```
- [ ] **Run the embedder**
  ```bash
  python embedder.py
  ```
  This will:
  - Read `data/processed_chunks.json`
  - Send batches of 500 texts to OpenAI `text-embedding-3-small`
  - Save results to `data/embedded_chunks.json` as a checkpoint
  - If interrupted, re-running will resume from where it left off (incremental)
- [ ] **Verify the checkpoint file**
  ```bash
  python3 -c "
  import json
  d = json.load(open('data/embedded_chunks.json'))
  print(f\"Model: {d['model']}\")
  print(f\"Dimensions: {d['dimensions']}\")
  print(f\"Chunks: {d['count']}\")
  print(f\"First vector length: {len(d['chunks'][0]['embedding'])}\")
  "
  ```
  Confirm: model is `text-embedding-3-small`, dimensions is `1536`, and every chunk has a 1536-element `embedding` array.

---

## Phase 4: Qdrant Setup & Upsert

- [ ] **Start a local Qdrant instance**
  ```bash
  docker run -d --name qdrant -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant
  ```
- [ ] **Write `upserter.py`** — script that:
  - Connects to Qdrant at `localhost:6333`
  - Creates (or recreates) collection `bills_2008_test` with `size=1536`, `distance=Cosine`
  - Creates payload indexes on `congress` (integer), `type` (keyword), `bill_id` (keyword)
  - Reads `data/embedded_chunks.json`
  - Upserts vectors in batches (100–500 per call) with payloads containing:
    - `bill_id`, `congress`, `type`, `number`, `short_title`
    - `section_enum`, `section_header`, `chunk_index`
    - `text` (the original chunk text, for display in search results)
  - Logs progress and final collection point count
- [ ] **Run the upserter**
  ```bash
  cd project-tarp
  python upserter.py
  ```
- [ ] **Verify collection health**
  ```bash
  python3 -c "
  from qdrant_client import QdrantClient
  c = QdrantClient('localhost', port=6333)
  info = c.get_collection('bills_2008_test')
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
- [ ] **Update `PLAN.md`** with actual results (chunk counts, cost, quality observations)
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
