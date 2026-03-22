---

kanban-plugin: board

---

## Backlog — Decisions

- [ ] Q1: Decide embedding model strategy (Ollama 768d vs OpenAI 1536d — collections are incompatible across dev/prod)
- [ ] Q2: Figure out Ollama networking in K8s (host process? NodePort? host.docker.internal?)
- [ ] Q3: Pick LLM provider for generation (Claude Sonnet prod / Ollama dev? abstract both?)
- [ ] Q4: Set up git hosting + ArgoCD remote (GitHub? self-hosted Gitea?)
- [ ] Q5: Storage backend — hostPath (SSD) for mars, defer NFS to netcup prod?
- [ ] Q6: Plan prod migration strategy (re-embed via API? snapshot sync from mars? proxy?)
- [ ] Q7: Vote data — separate collection, chunk type in bill_chunks, or defer?
- [ ] Q8: API auth + rate limiting for prod (LLM generation is expensive per-request)
- [ ] Q9: Cross-encoder memory on VPS — 500MB model + 1.5GB headroom enough under load?

## Backlog — Implementation

- [ ] Step 1: Infra foundations — namespace, PVs, secrets, ArgoCD, Qdrant StatefulSet, read-only PG role
- [ ] Step 2a: Data pipeline — fetcher (GovInfo XML bulk download w/ rate limiting + caching)
- [ ] Step 2b: Data pipeline — chunker (XML-aware section splitting, boilerplate filtering, context prepend)
- [ ] Step 2c: Data pipeline — batcher + upserter (batch embed via Ollama, stream vectors to Qdrant)
- [ ] Step 2d: Data pipeline — tracker (sync state so incremental runs skip already-embedded bills)
- [ ] Step 3a: RAG retrieval — query classifier (extract filters from natural language)
- [ ] Step 3b: RAG retrieval — embedder + dual retriever (Qdrant ANN + PG tsvector in parallel)
- [ ] Step 3c: RAG retrieval — RRF fusion + cross-encoder rerank
- [ ] Step 3d: RAG retrieval — metadata hydrator (full bill details from PG)
- [ ] Step 4a: API + generation — FastAPI server, routes, pydantic models, health checks
- [ ] Step 4b: API + generation — prompt builder + LLM streaming (SSE: sources → tokens → done)
- [ ] Step 4c: API + generation — Redis caching layer (embeddings 24h, results 1h, LLM 1h, metadata 6h)
- [ ] Step 4d: API + generation — Dockerize + push to private registry + K8s deploy
- [ ] Step 5: Nightly sync CronJob (00:30, incremental embed of ~100 new bills/day)
- [ ] Step 6: Frontend integration (SSE handling in Nuxt, keyword/NLP toggle)
- [ ] Step 7: Eval suite (100+ queries, recall@10, MRR, hallucination checks)

## In Progress



## Completed




%% kanban:settings
```
{"kanban-plugin":"board","date-colors":[{"distance":1,"unit":"days","direction":"after","isBefore":true,"color":"rgba(0, 119, 255, 1)"},{"distance":1,"unit":"days","direction":"after","isToday":true,"color":"rgba(255, 0, 0, 1)"},{"distance":1,"unit":"days","direction":"after","isAfter":true,"color":"rgba(255, 149, 0, 1)"}],"show-checkboxes":true,"new-card-insertion-method":"prepend"}
```
%%