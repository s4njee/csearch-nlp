# Project TARP — Congressional Votes Embeddings

## Objective
Extend the TARP semantic search pipeline to cover U.S. Congressional roll-call votes. Each vote is loaded from the `@unitedstates/congress` scraper's local data tree, normalized into a single embeddable text blob, embedded with OpenAI (`text-embedding-3-small`, 1536 dims), and upserted into PostgreSQL + pgvector alongside the existing `nlp.bill_chunks` / `nlp.bill_embeddings` tables. The query path reuses the existing OpenAI + pgvector stack and can optionally join a matched vote back to its referenced bill's chunks.

## Status
**Status:** Planned — not yet implemented.

---

## Why this is simpler than bills

Votes look superficially like bills but the pipeline shape is meaningfully smaller:

| Concern | Bills | Votes |
|---|---|---|
| Source | Remote (GovInfo XML + HTML fallback) | Local (scraper has already written `data.json`) |
| Text size | Multi-section legislative text, often >100 chunks/bill | A few short fields per vote, typically 1 embedding/vote |
| Parsing | XML `<section>` traversal, HTML regex fallback | Read JSON, concatenate a few fields |
| Chunking | Section-aware splitter with subsection fallback and token cap | Usually no split at all; overflow split only as a safety net |
| Dedup | Exact full-doc + exact section-body dedup (saves ~11K chunks/Congress) | Optional, low value — most procedural-question overlap is small and the embedding cost is negligible |
| Change detection | Content hasher (scraper refreshes XML attributes without text changes) | Not needed — once finalized a vote record is immutable |
| Scale | ~2.8M chunks across 50 years | ~1–2K votes/Congress × ~27 Congresses ≈ 30–60K votes total |
| Cost | ~$1 per Congress of bills | Well under $0.10 for the full historical corpus |

Design rule: **keep parallel structure with the bills pipeline (fetcher → chunker → embedder → upserter → query) so scripts, Docker images, and CronJob scaffolding can be reused, but delete the steps that don't earn their keep.**

---

## Tech Stack
Same as the bills pipeline:
- **Language**: Python 3.10+
- **Data Source**: `@unitedstates/congress` scraper output under `backend/scraper/congress/data/{congress}/votes/{session}/{chamber}{number}/data.json`
- **Embeddings API**: `openai` `text-embedding-3-small` (1536 dims, $0.02/1M tokens)
- **Generative API**: `openai` `gpt-5.4-nano` (for grounded answers in `query_votes.py`)
- **Vector DB**: PostgreSQL + pgvector, same database and `nlp` schema as bills
- **Token counting**: `tiktoken` (`cl100k_base`)

---

## Data source

Sample record at `backend/scraper/congress/data/119/votes/2025/h100/data.json`:

```json
{
  "vote_id": "h100-119.2025",
  "chamber": "h",
  "congress": 119,
  "session": "2025",
  "number": 100,
  "date": "2025-04-10T11:08:00-04:00",
  "category": "passage",
  "type": "Concurring in the Senate Amendment",
  "question": "On Motion to Concur in the Senate Amendment: H CON RES 14 ...",
  "subject": "Establishing the congressional budget for the United States Government for fiscal year 2025 ...",
  "result": "Passed",
  "requires": "1/2",
  "source_url": "https://clerk.house.gov/evs/2025/roll100.xml",
  "bill": { "congress": 119, "number": 14, "type": "hconres" },
  "votes": { "Yea": [...], "Nay": [...], "Present": [...], "Not Voting": [...] }
}
```

Free-text fields useful for semantic search: `question`, `subject`, `type`, `category`. Structured fields useful as payload / filters: `chamber`, `congress`, `session`, `date`, `result`, `bill.*`.

Scale from current scraper snapshot: 119th Congress ≈ 1,162 votes, 118th ≈ 1,932 votes. Earlier Congresses will be smaller per-year but span more sessions. Quorum calls and a minority of procedural votes have `bill: null`.

---

## Instructions for AI Agent

Execute sequentially. Each step should be functional and verified before moving on.

### Step 1: The Loader
**Goal:** Walk the scraper's votes tree and emit a normalized intermediate record per vote.
**Script:** `votes_loader.py`

**Responsibilities:**
- Walk `backend/scraper/congress/data/{congress}/votes/{session}/{chamber}{number}/data.json` for one or more congresses.
- For each vote, produce a normalized dict with:
  - `vote_id` (required — this is the canonical identity key)
  - `congress`, `session`, `chamber`, `number`
  - `date` (kept as the original ISO-8601 string; a separate `date_ts` for Postgres `timestamptz` is derived in the upserter)
  - `category`, `type`, `question`, `subject`, `result`, `requires`, `source_url`
  - `bill_id` flattened from the nested `bill` object when present (e.g. `{type: "hconres", number: 14, congress: 119}` → `hconres14-119`), `null` otherwise. Match the `bill_id` format that `chunker.py` already emits for bills so the join is trivial.
  - `source_path` (relative path to `data.json` from the repo root, useful for debugging)
- Write output as rolling JSONL shards under `./data/processed_votes/{congress}/shard-*.jsonl`, plus a per-Congress `manifest.json` with vote count and date range. Shard rotation at 2,000 records is fine — votes are small, a single shard per Congress is acceptable for historical loads.
- Do **not** include the per-legislator `votes.Yea/Nay/...` arrays in the embedding pipeline output. Those are already carried by the existing `vote_members` table populated by the scraper's Postgres loader and have no business being in the embedding index. They'd bloat rows by 10–50×.
- Skip records that have empty `question` AND empty `subject` (extremely rare — mostly malformed scraper files). Log a warning and continue.

**Incremental behaviour:** re-reading the tree is cheap (one JSON parse per file). No skip logic needed here — downstream steps are keyed on `vote_id` + content hash so re-processing an unchanged vote is a no-op.

### Step 2: The Chunker
**Goal:** Turn each normalized vote record into one (or rarely, multiple) chunk(s) ready for embedding.
**Script:** `votes_chunker.py`

**Why keep a chunker at all:** uniformity with the bills pipeline and a natural place to put the canonical embedding-text builder. 95%+ of votes will emit exactly one chunk.

**Embedding-text format** (this is the canonical string fed to OpenAI):
```
[Vote {vote_id}, {chamber_name}, {date_yyyy_mm_dd}] {category}: {type}
Question: {question}
Subject: {subject}
Result: {result}
{bill_line_if_present}
```

- `chamber_name` is `"House"` / `"Senate"` (map from `h`/`s`).
- `bill_line_if_present` is `"Related bill: {bill_id}"` when `bill_id` is set, omitted otherwise.
- Collapse whitespace and drop empty lines. If `subject` is a duplicate prefix of `question` (or vice versa), keep only the longer one — this happens often on House procedural votes.
- Result goes into `body`. The prefix `[Vote ...]` is deliberately part of the embedded text, matching the bills pipeline's decision to embed context with content.

**Per-chunk metadata emitted:**
- `vote_id`, `congress`, `chamber`, `session`, `number`, `date`, `category`, `type`, `result`, `bill_id`
- `token_count` (via `tiktoken`, `cl100k_base`)
- `chunk_index` — almost always 0; non-zero only when the vote's text exceeds `--max-tokens` (default 512) and we fall back to sentence-boundary splitting with `--overlap` (default 64). This is the same splitter used by `chunker.py` — factor it into a shared helper rather than copy-pasting.
- `content_hash` — SHA-256 of the normalized embedding-text body. Used by the embedder and upserter for identity.
- `source_hash` — SHA-256 of `vote_id` + `chunk_index` + `content_hash`. This is the stable row identity carried through to Postgres, matching the shape of the bills `source_hash`.

**No section-level dedup.** Votes are short and the procedural-question overlap across votes (e.g. "On Motion to Recommit") carries enough contextual difference (date, bill, chamber) in the embedded prefix that collapsing them would harm retrieval. Skip dedup entirely for the first version; revisit only if we see the same `content_hash` appearing thousands of times.

**Output:** mirrored JSONL shards at `./data/processed_vote_chunks/{congress}/shard-*.jsonl` plus a `manifest.json`.

### Step 3: The Embedder
**Goal:** Convert each chunk's `body` into a 1536-dim vector.
**Script:** reuse `embedder.py` if possible — parameterize the input/output directory defaults.

**Required changes to the shared embedder:**
- Accept `--input-dir` / `--output-dir` overrides so it can run against `processed_vote_chunks/` → `embedded_vote_chunks/`.
- Identity for shard-local checkpointing becomes `source_hash` (already unique per vote chunk). This is equivalent to what the bills embedder already does using its canonical chunk identity; the shared abstraction is "whatever the `source_hash` field says."
- `--dry-run` prints cost estimate using the chunk token counts.

**Expected cost:** historical corpus ~60K votes × ~120 tokens each ≈ 7.2M tokens × $0.02/1M = **~$0.15 one-time**, negligible recurring.

### Step 4: Storage (PostgreSQL Upsert)
**Goal:** Load vectors into pgvector in the same database as bills.
**Script:** `votes_upserter.py` (or extend `upserter.py` with a `--mode {bills,votes}` flag — prefer this since ~80% of the DDL/index/upsert code is identical).

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS nlp.vote_chunks (
  id            BIGSERIAL PRIMARY KEY,
  source_hash   TEXT NOT NULL UNIQUE,
  vote_id       TEXT NOT NULL,
  congress      INTEGER NOT NULL,
  chamber       TEXT NOT NULL,
  session       TEXT NOT NULL,
  number        INTEGER NOT NULL,
  vote_date     TIMESTAMPTZ,
  category      TEXT,
  vote_type     TEXT,
  question      TEXT,
  subject       TEXT,
  result        TEXT,
  bill_id       TEXT,            -- nullable, references the bill family but no FK
  body          TEXT NOT NULL,   -- the embedded text
  token_count   INTEGER NOT NULL,
  chunk_index   INTEGER NOT NULL DEFAULT 0,
  content_hash  TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nlp.vote_embeddings (
  chunk_id   BIGINT PRIMARY KEY REFERENCES nlp.vote_chunks(id) ON DELETE CASCADE,
  embedding  vector(1536) NOT NULL,
  model      TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS vote_chunks_vote_id_idx  ON nlp.vote_chunks (vote_id);
CREATE INDEX IF NOT EXISTS vote_chunks_bill_id_idx  ON nlp.vote_chunks (bill_id);
CREATE INDEX IF NOT EXISTS vote_chunks_congress_idx ON nlp.vote_chunks (congress);
CREATE INDEX IF NOT EXISTS vote_chunks_chamber_idx  ON nlp.vote_chunks (chamber);
CREATE INDEX IF NOT EXISTS vote_chunks_date_idx     ON nlp.vote_chunks (vote_date);

-- HNSW index built in the final pass (same as bills)
CREATE INDEX vote_embeddings_embedding_hnsw_idx
  ON nlp.vote_embeddings USING hnsw (embedding vector_cosine_ops)
  WITH (m=16, ef_construction=128);
```

**Notes:**
- `bill_id` is intentionally not a foreign key. The bills pipeline works in terms of `canonical_bill_id` (collapsed duplicates) and `bill_id` in votes is the raw scraper form; maintaining referential integrity across the two would force the votes pipeline to wait for the bills pipeline, which we don't want. Instead, the query layer LEFT JOINs on `bill_id` with a normalization function and tolerates misses.
- Do not add a FK to any scraper-owned `votes` table either — the scraper's Postgres loader is a sibling pipeline and we want the vector side to be independently reloadable.
- Same `--skip-hnsw` / `--index-only` flow as bills for bulk loads.
- Same `--recreate` semantics.

**Upsert behaviour:** keyed on `source_hash`. An unchanged vote is a no-op; a changed one (shouldn't happen, but the scraper occasionally backfills) deletes the old row and re-inserts.

### Step 5: The Query Engine
**Goal:** Semantic search across votes, with optional join into bill excerpts for grounded answers.
**Script:** `query_votes.py` — prefer extending `query.py` with a `--index {bills,votes,both}` flag.

**Query behaviour:**
- Embed the user query with `text-embedding-3-small`.
- Top-k search `nlp.vote_embeddings` with cosine distance.
- For each hit, print: `vote_id`, date, chamber, result, question (truncated), and `bill_id` if set.
- In `--index both` mode, run a parallel search over `nlp.bill_embeddings` and interleave results by distance, annotating the source.
- Optional grounded answer: pass the top hits to `gpt-5.4-nano` with a prompt like "Given these Congressional votes (and related bill excerpts if present), answer the user's question. Cite `vote_id` and `bill_id` inline."
- When a hit has a `bill_id`, the answer-generation step may do a secondary vector search inside `nlp.bill_embeddings` filtered to that `bill_id` to pull in one or two bill section excerpts for context. This is where the cross-index linkage pays off.

**Sample queries to validate with:**
- "How did the House vote on the 2008 financial bailout?"
- "Recent Senate votes on immigration"
- "Budget reconciliation votes in the 119th Congress"

---

## Incremental update pipeline

Mirror `nightly_update.sh` as `nightly_update_votes.sh`, or (preferred) extend the existing script with a `VOTES=1` switch so one CronJob handles both. The pipeline shape:

| Script | Incremental behaviour |
|---|---|
| `votes_loader.py` | Full rewalk per run. Cheap — filesystem walk + JSON parse for ~1K new files/night max. |
| `votes_chunker.py` | Full rewrite per Congress. CPU-only, seconds. |
| `embedder.py` (shared) | Skips chunks whose `source_hash` already exists in the embedded shard manifest. Only new votes cost API money. |
| `upserter.py --mode votes` | Idempotent per `source_hash`. `--skip-hnsw` leaves index maintenance to pgvector's incremental updates. |

No `content_hasher.py` equivalent — votes don't get XML-attribute churn the way bill XMLs do.

---

## Repository layout (new/changed files)

```
project-tarp/
  votes_loader.py                # NEW — walks scraper data, writes processed_votes shards
  votes_chunker.py               # NEW — builds embedding-text + per-chunk metadata
  embedder.py                    # MODIFIED — accepts --input-dir / --output-dir
  upserter.py                    # MODIFIED — --mode {bills,votes} branches on table/DDL
  query.py                       # MODIFIED — --index {bills,votes,both}
  nightly_update.sh              # MODIFIED — runs both pipelines in sequence
  Dockerfile.nightly-updater     # REBUILD — same base image, just picks up new scripts

k8s/
  nlp-nightly-updater-cronjob.yaml  # no change needed if the updater script handles both
```

The Dockerfile does not need a new image — the votes scripts ship in the same updater image.

---

## Cost Estimate

`text-embedding-3-small` at $0.02/1M tokens.

| Scenario | Est. tokens | Est. cost |
|---|---|---|
| One Congress of votes (~1,500 votes × ~120 tokens) | 180K | ~$0.004 |
| Full historical backfill (~60K votes) | 7.2M | ~$0.15 |
| Typical nightly delta (0–50 new votes) | 0–6K | $0.00–$0.0001 |

Storage: 60K votes × ~6KB per row (including vector) ≈ **~360 MB total** in Postgres. The HNSW index on 60K vectors is negligible (~50 MB).

---

## Open decisions to revisit after first run

- **Embedding-text inclusion of `result`**: including "Passed" / "Failed" may help queries like "bills that failed to pass on immigration" but also may bias similarity toward outcome rather than topic. Try both; measure with a handful of eval queries.
- **Per-legislator signal**: out of scope here, but a future extension could embed a compact "voting coalition signature" per vote (e.g. hash of Yea voters) to support queries like "votes where the same coalition voted together." That belongs in a separate index, not in `vote_chunks`.
- **Companion-bill boost at query time**: when a vote's top hit has a `bill_id`, should we pull the bill's top chunks into the answer prompt automatically, or only on a follow-up "tell me more"? Default to automatic for `--index both`, off otherwise.
- **Near-duplicate procedural votes**: if post-launch analytics show the same procedural `question` text (e.g. motions to recommit) saturating result sets, add an optional dedup pass that collapses by normalized question+chamber+congress and keeps the latest `vote_id`. Don't build this preemptively.
