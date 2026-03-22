# Project TARP — 2008 LLM Embeddings Proof of Concept

## Objective
Build a minimal, end-to-end pipeline to download all U.S. Congressional bills from 2008 (the 110th Congress), clean and chunk their text, generate vector embeddings using OpenAI (`text-embedding-3-small`), store them in a local Qdrant instance, and run fundamental semantic search queries against the database. 

This is a proof-of-concept (POC) to validate the chunking strategy and OpenAI embedding costs (~$1.00 total) before scaling to the full 50-year dataset in the main `csearch-nlp` repository.

## Tech Stack
- **Language**: Python 3.10+
- **Data Source**: GovInfo Bulk Data API (`/BILLS/110`)
- **XML Parsing**: `lxml` or `BeautifulSoup4`
- **Token Counting**: `tiktoken`
- **Embeddings API**: `openai` (using `text-embedding-3-small` to convert text chunks into vector arrays)
- **Generative API**: `openai` (using `gpt-5.4-nano` to synthesize final answers)
- **Vector DB**: `qdrant-client` (running Qdrant via local Docker container)

---

## Instructions for AI Agent

**Agent Directive:** Execute the following steps sequentially. Do not proceed to the next step until the current step is fully functional and tested.

### Step 1: The Fetcher (GovInfo XML Downloader)
**Goal:** Download the raw `.xml` bill files for the 110th Congress.
- Write a Python script (`fetcher.py`).
- Hit the GovInfo Bulk Data API for the 110th Congress (`https://www.govinfo.gov/bulkdata/BILLS/110/{billtype}/`).
- Filter and download only the latest version of House (`HR`) and Senate (`S`) bills available.
- Save the downloaded XML files to a local directory: `./data/bills_110/`.
- Ensure the script has basic error handling and rate-limiting (e.g., small sleeps) to comply with GovInfo API limits.

### Step 2: The XML Parser & Chunker
**Goal:** Parse the legal jargon into semantically meaningful chunks with context prepended. 
- Write a Python script (`chunker.py`).
- Iterate over the downloaded XML files in `./data/bills_110/`.
- **Extraction**: Extract metadata from the XML: `bill_id` (e.g., hr1424-110), `congress` (110), `year` (2007 or 2008), and `short_title`.
- **Targeting**: Traverse the XML tree and isolate `<section>` tags.
- **Filtering**: Discard boilerplate sections entirely (e.g., "Effective Date", "Severability", "Short Title").
- **Context Assembly**: For valid sections, reconstruct the text and prepend the bill's context string format: `"[H.R. 1424, 110th Congress] Section 101: Purchases of Troubled Assets — {Actual Section Text}"`.
- **Token Limits**: Use the `tiktoken` library to encode the text using OpenAI's tokenizer. If a section exceeds ~512 tokens, split it at a structural boundary (like `<paragraph>`) with a 64-token overlap between chunks.
- Output the finalized chunks alongside their metadata into a structured JSON file (e.g., `./data/processed_chunks.json`).

### Step 3: The Embedder (OpenAI Integration)
**Goal:** Convert text chunks into 1536-dimensional vector arrays.
- Write a Python script (`embedder.py`).
- Read the structured data from `./data/processed_chunks.json`.
- Group the text strings into batches (e.g., batches of 500-1000 strings).
- Call `openai.embeddings.create(input=batch, model="text-embedding-3-small")`.
- Append the returned 1536-dimensional float arrays back into the chunk data records.
- **Crucial Requirement:** Save the embedded data strictly as a checkpoint (`./data/embedded_chunks.json`) so the API call doesn't need to be re-run (costing money) during Step 4 debugging.

### Step 4: Storage Setup (Qdrant Upsert)
**Goal:** Load the vectors and metadata into a local Vector DB.
- Write a Python script (`upserter.py`).
- Ensure Qdrant is running locally via Docker (`docker run -p 6333:6333 qdrant/qdrant`).
- Use the `qdrant-client` library in Python to connect to `localhost:6333`.
- Recreate a collection named `bills_2008_test` with `size=1536` and `distance=Cosine`.
- Load the checkpointed data from Step 3.
- Map the structures to Qdrant's expected format (ID, Vector array, and Payload/Metadata dictionary).
- Use `client.upsert()` to ingest the data in batches into the Qdrant collection.

### Step 5: The Query Engine & Answer Generation
**Goal:** Run semantic searches against the data and generate readable answers for the user using OpenAI's latest models.
- Write a minimal Python script (`query.py`).
- Ask the user for textual input via the terminal (e.g., `user_query = input("Enter search query: ")`).
- Immediately pass `user_query` through the OpenAI Embedding API (`text-embedding-3-small`) to retrieve its single 1536-dimensional embedding.
- Pass this query vector to Qdrant's `client.search()` function targeting the `bills_2008_test` collection to retrieve the top 5 closest matching chunks.
- Pass the textual payload of those top 5 chunks into the new generative model (`gpt-5.4-nano`) to write a cohesive, summarized answer citing its sources.
- Print out the generated response to the console.
