# Project TARP — 2008 Bill Embeddings Proof of Concept

## Objective
Build a minimal, end-to-end pipeline to download all U.S. Congressional bills from the 110th Congress (2007–2008), clean and chunk their text, generate vector embeddings using OpenAI (`text-embedding-3-small`), store them in PostgreSQL with `pgvector`, and run semantic search queries with LLM-generated answers.

This is a proof-of-concept (POC) to validate the chunking strategy and OpenAI embedding costs before scaling to the full 50-year dataset in the main `csearch-nlp` repository.

## POC Status
**Status:** ✅ End-to-end POC complete

The Project TARP pipeline now runs end to end for the 110th Congress:

- raw bill text fetched from GovInfo
- bill text parsed and chunked with exact section-level dedup
- full corpus embedded with OpenAI `text-embedding-3-small`
- vectors loaded into PostgreSQL `pgvector` tables on the `mars` Kubernetes cluster
- semantic search and grounded answer generation working through `query.py`

Representative observed outputs:

- chunking: `439,890` canonical chunks across `21` shards
- dedup: `11,571` duplicate sections removed
- embedding tokens: `55,260,842`
- full-corpus embedding cost: about `$1.11`
- PostgreSQL DSN: `PG_CONNECTION_STRING`

## Tech Stack
- **Language**: Python 3.10+
- **Data Source**: GovInfo content API + `@unitedstates/congress` scraper metadata
- **XML Parsing**: `xml.etree.ElementTree` (stdlib)
- **Token Counting**: `tiktoken` (`cl100k_base` encoding, used by `text-embedding-3-small`)
- **Embeddings API**: `openai` (`text-embedding-3-small`, 1536 dims, $0.02/1M tokens)
- **Generative API**: `openai` (`gpt-5.4-nano`, $0.20/1M input tokens)
- **Vector DB**: PostgreSQL + `pgvector` (`psycopg2-binary`)

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

### Step 2: The XML Parser & Chunker ✅ COMPLETE
**Goal:** Parse bill text into semantically meaningful chunks with context prepended.
**Script:** `chunker.py` (569 lines)
**Status:** Full 110th Congress chunking run completed. Current production run with `--max-chunks-per-bill 300` produced 439,890 canonical chunks across 21 JSONL shards from 10,493 canonical bills. Dedup removed 11,571 duplicate sections; exact full-document dedup found 0 duplicate documents in this corpus.

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
- Per-bill chunk cap (`--max-chunks-per-bill`, default 200) is section-aware: it first keeps the strongest chunk from each section, then fills the remaining budget with the strongest leftover chunks bill-wide.
- Filters out chunks below 30 tokens.
- Outputs rolling JSONL shards under `./data/processed_chunks/{congress}/` rather than one monolithic JSON file. Each line is one canonical chunk record with dedup metadata. Each chunk should carry at minimum: `document_text_hash`, `section_text_hash` (when applicable), `canonical_bill_id`, and alias metadata describing which bills/sections map to that text.
- Sharding is by canonical bill count with a soft chunk cap, not by bill type. This avoids giant `hr`/`s` shards while keeping all chunks for a canonical bill together.
- Default shard policy: rotate after 500 canonical bills or 40,000 chunks, but only flush between canonical bills.
- Each congress directory also gets a `manifest.json` with shard counts, chunk counts, and dedup summary stats.
- After the per-bill cap is applied, surviving chunks are renumbered to contiguous `chunk_index` values within each canonical section; the pre-cap index is preserved as `original_chunk_index`.
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
- The cap policy now preserves section coverage by keeping one best chunk per section before filling the remaining budget. An alternative is a pure top-token global cap, which is simpler but can erase small sections entirely.

### Step 3: The Embedder ✅ COMPLETE
**Goal:** Convert text chunks into 1536-dimensional vector arrays via OpenAI API.
**Script:** `embedder.py` (204 lines)
**Status:** Full embedding run completed against the sharded chunk output. The embedder now writes mirrored embedded JSONL shards under `./data/embedded_chunks/` with shard-local checkpointing and retry/backoff for rate limits.

**Key implementation details:**
- Reads `./data/processed_chunks/` recursively, loading all `shard-*.jsonl` files, then sends batches of texts to `openai.embeddings.create()`.
- Writes embedded output as mirrored JSONL shards under `./data/embedded_chunks/` rather than one giant JSON checkpoint file. Example: `processed_chunks/110/shard-00003.jsonl` maps to `embedded_chunks/110/shard-00003.jsonl`.
- Incremental: if an embedded shard already exists, only missing chunks in that shard are sent to the API. Checkpoint identity is based on canonical chunk identity (`document_text_hash` + `section_text_hash` + `chunk_index`) rather than raw `bill_id` alone.
- Checkpoint saves are shard-local and periodic, so resume is cheap and does not require rewriting a monolithic file.
- `--dry-run` flag shows cost estimate without calling the API.
- Output format: one embedded chunk per JSONL line, with the original chunk metadata plus an `"embedding"` field containing the 1536-float vector. The embedded shard directory also gets a lightweight `manifest.json`.
- Supports `--model` and `--dimensions` flags for experimenting with `text-embedding-3-large`.

### Step 4: Storage Setup (PostgreSQL pgvector Upsert) ✅ COMPLETE
**Goal:** Load the vectors and metadata into PostgreSQL tables backed by `pgvector`.
- `upserter.py` is implemented and reads embedded shard files from `./data/embedded_chunks/`.
- The loader targets `nlp.bill_chunks` and `nlp.bill_embeddings` via `PG_CONNECTION_STRING`.
- The database has the `vector` extension enabled and uses `vector(1536)` columns plus an HNSW index for similarity search.
- Upserts use deterministic chunk source hashes so reruns remain idempotent.
- Payloads are intentionally lean by default; large alias arrays are omitted unless explicitly requested.

### Step 5: The Query Engine & Answer Generation ✅ COMPLETE
**Goal:** Run semantic searches and generate readable answers using OpenAI's latest models.
- `query.py` is implemented as an interactive CLI over PostgreSQL `pgvector`.
- The query path embeds the user query with `text-embedding-3-small`, searches `nlp.bill_embeddings` joined to `nlp.bill_chunks`, prints the top semantic matches, and optionally asks `gpt-5.4-nano` for a grounded answer.
- The first end-to-end test query was successful:
  - Query: `"What did Congress do about the financial crisis in 2008?"`
  - Top hits included `hr7275-110`, `hr7104-110`, and `hr3666-110`
  - The generated answer correctly summarized the retrieved excerpts as proposed commissions/investigations plus foreclosure-related findings, while explicitly noting uncertainty where the retrieved excerpts did not show enacted remedies.
- A sample run is recorded in `SAMPLE.md`.
