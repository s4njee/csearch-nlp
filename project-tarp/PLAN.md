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
- Deduplication happens before embedding and is part of the chunking pipeline, not a later storage concern.
- **Phase 1 — exact full-document dedup:** normalize each fetched bill's full text (strip markup, collapse whitespace, lowercase for hashing only), compute a `document_text_hash`, and collapse bills with byte-for-byte equivalent normalized text into one canonical document record.
- Preserve bill identity even when text is deduplicated: each canonical document keeps an alias list containing every linked `bill_id`, `version`, `status`, `congress`, `type`, and `number` that resolved to that same text.
- Canonical document selection should be deterministic. Prefer the alias with the "best" version in this order: `enr` > `eas`/`eah` > `es`/`eh` > `rs`/`rh` > `ih`/`is`; break remaining ties by `bill_id`.
- Chunk only the canonical document text. Do not emit duplicate chunk rows for alias bills whose full text is identical.
- **Phase 2 — exact section-level dedup:** after parsing a canonical document into sections, normalize each section body separately and compute a `section_text_hash`. If the same section text appears multiple times across canonical documents, embed it once and attach multiple section aliases to it.
- Section-level dedup must hash the substantive section body, not the bill-specific context prefix. The current prefix format includes bill identifiers, so hashing the final chunk text would miss duplicates across companion bills and status aliases.
- Keep legally meaningful differences. Dedup only exact normalized matches; do not merge near-duplicates or fuzzy companion bills at this stage.
- Two parsing paths: XML bills (majority, uses `<section>` tag traversal) and HTML/text bills (fallback, regex-based `SECTION`/`SEC.` splitting).
- Extracts metadata from both the XML structure (Dublin Core, `<form>` element) and the fetcher's `.meta.json` sidecar files.
- Context prefix format: `"[H.R. 1424, 110th Congress] Section 101: Purchases of Troubled Assets — {text}"`.
- Boilerplate filtering: skips sections titled "Short Title", "Effective Date", "Severability", "Table of Contents".
- Token counting via `tiktoken` (`cl100k_base`), with word-count fallback if tiktoken is unavailable.
- Sections exceeding `--max-tokens` (default 512) are split at `<subsection>` boundaries first, then force-split with `--overlap` (default 64) token overlap at sentence boundaries.
- Per-bill chunk cap (`--max-chunks-per-bill`, default 200) keeps the meatiest chunks and drops the rest.
- Filters out chunks below 30 tokens.
- Outputs rolling JSONL shards under `./data/processed_chunks/{congress}/` rather than one monolithic JSON file. Each line is one canonical chunk record with dedup metadata. Each chunk should carry at minimum: `document_text_hash`, `section_text_hash` (when applicable), `canonical_bill_id`, and alias metadata describing which bills/sections map to that text.
- Sharding is by canonical bill count with a soft chunk cap, not by bill type. This avoids giant `hr`/`s` shards while keeping all chunks for a canonical bill together.
- Default shard policy: rotate after 500 canonical bills or 40,000 chunks, but only flush between canonical bills.
- Each congress directory also gets a `manifest.json` with shard counts, chunk counts, and dedup summary stats.
- Reports estimated embedding cost at the end of the run.

**Deduplication data model:**
- `canonical_bill_id`: stable bill identifier chosen to represent an exact full-text-equivalent bill family.
- `document_text_hash`: normalized full-document hash used for Phase 1 exact dedup.
- `section_text_hash`: normalized section-body hash used for Phase 2 exact dedup across canonical documents.
- `document_aliases`: list of all bill/status/version records that share the canonical document text.
- `section_aliases`: list of all canonical bill sections that share the exact same normalized section body.
- `manifest.json`: congress-level summary of shard layout plus dedup and chunk counts.

**Pipeline placement:**
1. Fetch one best-available text per bill with `fetcher.py`.
2. Normalize full text and collapse exact duplicate documents into canonical records.
3. Parse only canonical documents into sections.
4. Normalize section bodies and collapse exact duplicate sections across canonical documents.
5. Build chunks from canonical section bodies, then add display-time prefixes and alias metadata.
6. Write deduplicated canonical chunks into rolling JSONL shards and a congress manifest.
7. Pass the shard directory to the embedder.

**Expected effect:**
- Removes waste from bills that were reintroduced or progressed through statuses without textual change.
- Reduces repeated sections shared across companion bills or unchanged later versions.
- Keeps retrieval quality intact because aliases are preserved for display, filtering, and downstream reranking.

**Implementation notes / alternative decisions:**
- Full-document dedup hashes normalized extracted text from the fetched file. An alternative would be hashing parsed section text only, which would ignore front matter differences but could accidentally merge documents that differ outside the parsed section set.
- Section-level dedup hashes the section body only, excluding bill-specific prefixes and section numbering. An alternative would include headers or enum labels in the hash, which is safer but would miss duplicates where only numbering changed.
- Canonical chunks currently use the highest-priority alias's bill prefix for embedding text and keep all other bill/status mappings in `document_aliases` and `section_aliases`. An alternative would be to embed a neutralized prefix-free body and add bill context only at retrieval time.
- Dedup remains exact-match only. A future alternative is near-duplicate clustering for companion bills, but that should be treated as a retrieval/display optimization rather than a hard drop from the embedding corpus.
- Shards are based on canonical bill count plus a soft chunk ceiling. An alternative is fixed-size chunk-only sharding, which balances file sizes more tightly but makes it easier to split one bill across shards.

### Step 3: The Embedder ✅ CODE COMPLETE (not yet run)
**Goal:** Convert text chunks into 1536-dimensional vector arrays via OpenAI API.
**Script:** `embedder.py` (204 lines)
**Status:** Code complete. Waiting on the full chunker output before running (to avoid paying for a partial dataset).

**Key implementation details:**
- Reads `./data/processed_chunks/` recursively, loading all `shard-*.jsonl` files, then sends batches of 500 texts to `openai.embeddings.create()`.
- Incremental: if `./data/embedded_chunks.json` already exists, only embeds chunks not already in the checkpoint. After dedup lands, checkpoint identity should be based on canonical chunk identity (`document_text_hash` + `section_text_hash` + `chunk_index`) rather than raw `bill_id` alone.
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
