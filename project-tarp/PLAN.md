# Project TARP — 2008 Bill Embeddings Proof of Concept

## Objective
Build a minimal, end-to-end pipeline to download all U.S. Congressional bills from the 110th Congress (2007–2008), clean and chunk their text, generate vector embeddings using OpenAI (`text-embedding-3-small`), store them in a local Qdrant instance, and run semantic search queries with LLM-generated answers.

This is a proof-of-concept (POC) to validate the chunking strategy and OpenAI embedding costs before scaling to the full 50-year dataset in the main `csearch-nlp` repository.

## Tech Stack
- **Language**: Python 3.10+
- **Data Source**: GovInfo content API + `@unitedstates/congress` scraper metadata
- **XML Parsing**: `xml.etree.ElementTree` (stdlib)
- **Token Counting**: `tiktoken` (`cl100k_base` encoding, used by `text-embedding-3-small`)
- **Embeddings API**: `openai` (`text-embedding-3-small`, 1536 dims, $0.02/1M tokens)
- **Generative API**: `openai` (`gpt-5.4-nano`, $0.20/1M input tokens)
- **Vector DB**: `qdrant-client` (running Qdrant via local Docker container)

## Data Source
Bill metadata comes from the `@unitedstates/congress` scraper, which populates a local `110/bills/{type}/{type}{number}/data.json` directory tree at the project root. The fetcher reads this metadata (bill status, title, etc.) to determine which GovInfo version to download (enrolled > engrossed > reported > introduced), then saves the full text to `project-tarp/data/bills_110/{type}/`.

---

## Instructions for AI Agent

**Agent Directive:** Execute the following steps sequentially. Do not proceed to the next step until the current step is fully functional and tested. See `ToDo.md` for detailed status tracking and verification commands.

### Step 1: The Fetcher ✅ COMPLETE
**Goal:** Download the raw bill text files for the 110th Congress.
**Script:** `fetcher.py` (383 lines)
**Status:** 7,960 bills discovered across all bill types (HR, S, HJRES, SJRES, etc.). 7,789 XML files successfully downloaded to `./data/bills_110/`. Each bill also has a `.meta.json` sidecar with source URL, version, format, and title metadata.

**Key implementation details:**
- Uses the `@unitedstates/congress` scraper's `data.json` files for bill metadata and status.
- Constructs GovInfo content URLs per bill (`https://www.govinfo.gov/content/pkg/BILLS-{congress}{type}{number}{version}/xml/...`).
- Prioritizes versions based on bill status (enacted bills try `enr` first, introduced bills try `ih`/`is`).
- Handles GovInfo's soft-404s (HTTP 200 with error page body) by scanning response content for error signatures.
- Rate-limited at ~3 req/s with 10s backoff on 429 responses.
- Idempotent: skips bills where `.meta.json` already exists.
- Supports `--clean` mode to remove previously saved error pages.
- Falls back through XML → HTML → plain text formats per bill.

### Step 2: The XML Parser & Chunker ✅ CODE COMPLETE (needs full run)
**Goal:** Parse bill text into semantically meaningful chunks with context prepended.
**Script:** `chunker.py` (569 lines)
**Status:** Tested on 20 bills → 432 chunks. Needs to be run on the full 7,789-bill dataset.

**Key implementation details:**
- Two parsing paths: XML bills (majority, uses `<section>` tag traversal) and HTML/text bills (fallback, regex-based `SECTION`/`SEC.` splitting).
- Extracts metadata from both the XML structure (Dublin Core, `<form>` element) and the fetcher's `.meta.json` sidecar files.
- Context prefix format: `"[H.R. 1424, 110th Congress] Section 101: Purchases of Troubled Assets — {text}"`.
- Boilerplate filtering: skips sections titled "Short Title", "Effective Date", "Severability", "Table of Contents".
- Token counting via `tiktoken` (`cl100k_base`), with word-count fallback if tiktoken is unavailable.
- Sections exceeding `--max-tokens` (default 512) are split at `<subsection>` boundaries first, then force-split with `--overlap` (default 64) token overlap at sentence boundaries.
- Per-bill chunk cap (`--max-chunks-per-bill`, default 200) keeps the meatiest chunks and drops the rest.
- Filters out chunks below 30 tokens.
- Outputs `./data/processed_chunks.json` with full metadata per chunk.
- Reports estimated embedding cost at the end of the run.

### Step 3: The Embedder ✅ CODE COMPLETE (not yet run)
**Goal:** Convert text chunks into 1536-dimensional vector arrays via OpenAI API.
**Script:** `embedder.py` (204 lines)
**Status:** Code complete. Waiting on the full chunker output before running (to avoid paying for a partial dataset).

**Key implementation details:**
- Reads `./data/processed_chunks.json`, sends batches of 500 texts to `openai.embeddings.create()`.
- Incremental: if `./data/embedded_chunks.json` already exists, only embeds chunks not already in the checkpoint (keyed by `bill_id` + `section_enum` + `chunk_index`).
- Checkpoint saves after each batch on error, so a failed run can resume without re-paying.
- `--dry-run` flag shows cost estimate without calling the API.
- Output format: `{"model": "...", "dimensions": 1536, "count": N, "chunks": [...]}` where each chunk has an `"embedding"` field containing the 1536-float vector.
- Supports `--model` and `--dimensions` flags for experimenting with `text-embedding-3-large`.

### Step 4: Storage Setup (Qdrant Upsert) — NOT STARTED
**Goal:** Load the vectors and metadata into a local Qdrant instance.
- Write a Python script (`upserter.py`).
- Start Qdrant locally via Docker: `docker run -d -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant`
- Use the `qdrant-client` library to connect to `localhost:6333`.
- Create a collection named `bills_2008_test` with `size=1536` and `distance=Cosine`.
- Create payload indexes: `congress` (integer), `type` (keyword), `bill_id` (keyword).
- Read the checkpoint from `./data/embedded_chunks.json`.
- Upsert vectors in batches (100–500 per call) with payload metadata: `bill_id`, `congress`, `type`, `number`, `short_title`, `section_enum`, `section_header`, `chunk_index`, and `text`.
- Log progress and verify final collection point count matches chunk count.

### Step 5: The Query Engine & Answer Generation — NOT STARTED
**Goal:** Run semantic searches and generate readable answers using OpenAI's latest models.
- Write a Python script (`query.py`).
- Accept a natural language query from stdin.
- Embed the query via `text-embedding-3-small` (single API call, cost: ~$0.000002).
- Search Qdrant `bills_2008_test` for the top 5 closest vectors.
- Display matched chunks: score, bill ID, section header, and a text snippet.
- Pass the top 5 chunks as context to `gpt-5.4-nano` with a system prompt instructing it to cite specific bill numbers, quote statutory language, and never fabricate.
- Print the generated answer to the terminal.
- **Test queries to validate:**
  - "What did Congress do about the financial crisis in 2008?"
  - "bills about bank bailouts"
  - "legislation regulating subprime mortgages"
  - "environmental protection bills from the 110th Congress"
  - "bills related to veterans healthcare"
