**Overview**
- **What:** A lightweight ingestion + enrichment pipeline and a polling ingestion agent for `GraphIQ/data`.
- **Why:** Parse documents (PDF/DOCX/HTML/TXT), chunk them, generate summaries, optionally embed & index, and store minimal chunk records.

**What I changed (recently)**
- Moved the runnable pipeline into `GraphIQ/scripts/ingest_enrich_pipeline.py` and kept a small wrapper at `GraphIQ/ingest_enrich_pipeline.py` for backward compatibility.
- Implemented a safe loader in `GraphIQ/scripts/ingest_agent.py` so importing the pipeline does not trigger CLI argument parsing.
- Added a migration helper `GraphIQ/scripts/migrate_chunks_minimal.py` to back up and rewrite `chunks.jsonl` to the requested minimal schema.

**File layout (important files)**
- `GraphIQ/scripts/ingest_enrich_pipeline.py`: core pipeline (parsers, chunker, summarizer, embedder, storage). Use this for direct one-off runs.
- `GraphIQ/scripts/ingest_agent.py`: polling agent that watches a data directory and calls the pipeline on new/changed files.
- `GraphIQ/scripts/migrate_chunks_minimal.py`: migration tool that backs up and rewrites `chunks.jsonl` keeping only `chunk_id`, `text`, and `summary`.
- `GraphIQ/data/output/chunks.jsonl`: output JSONL produced by the pipeline (one JSON object per line).

**How the pipeline writes chunks (current behavior)**
- Each saved chunk line is a JSON object with these keys: `chunk_id` (string), `text`, `summary`.
- `chunk_id` is currently generated from the chunk index (stringified index). If you want global uniqueness across files, we can prefix with filename.

**Run commands**
- One-shot pipeline on a single file or a directory (will iterate supported files):
```bash
python GraphIQ/scripts/ingest_enrich_pipeline.py --input GraphIQ/data --outdir GraphIQ/data/output
```

- Run the polling agent (watches `--data-dir` and writes to `--outdir`):
```bash
python GraphIQ/scripts/ingest_agent.py --data-dir GraphIQ/data --outdir GraphIQ/data/output
```

- Migrate existing `chunks.jsonl` to the minimal format (creates a timestamped backup automatically):
```bash
python GraphIQ/scripts/migrate_chunks_minimal.py --chunks GraphIQ/data/output/chunks.jsonl
```

- Preserve original objects (including embeddings) when migrating:
```bash
python GraphIQ/scripts/migrate_chunks_minimal.py --preserve-embeddings --chunks GraphIQ/data/output/chunks.jsonl
```

**Notes & suggestions**
- Backups: `migrate_chunks_minimal.py` copies the original file to `chunks.jsonl.bak.<TIMESTAMP>` before rewriting.
- Embeddings: the migration strips embeddings from `chunks.jsonl`; use `--preserve-embeddings` to keep a copy (`chunks_with_embeddings.jsonl`).
- PDF support: parsing PDFs requires `pdfplumber`. If not installed, PDFs are skipped and a warning is logged.
- Unique chunk ids: if you'd like `chunk_id` to be globally unique, I can change the pipeline to prefix `chunk_id` with filename and/or a UUID.

**If you want me to proceed**
- I can run the migration now and show the first 10 lines of the resulting `chunks.jsonl` to confirm the format.
- Or I can change `chunk_id` generation to include filename prefixes before migrating.

**Contact / next steps**
- Tell me whether to run the migration now and whether to preserve embeddings or change `chunk_id` format.

---
Generated: automated README for the ingest/enrich work (27 Nov 2025)
