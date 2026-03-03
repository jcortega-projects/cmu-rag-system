# CMU Pittsburgh-Centric RAG Pipeline

## Project Overview
This repository contains a custom Retrieval-Augmented Generation (RAG) system for CMU Advanced NLP, focused on hyper-local and temporal QA about Pittsburgh and Carnegie Mellon University. The pipeline includes multi-source ingestion, chunking, dense + sparse indexing, configurable retrieval mode selection (`dense`, `sparse`, `hybrid`), and local Llama 3.1 generation.

## Repository Structure
```text
config/
└── urls.json

src/
├── __init__.py
├── main.py
├── pipeline.py
├── ingestion/
│   └── data_ingestion.py
├── chunking/
│   └── text_splitter.py
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   └── searcher.py
└── generation/
    └── reader.py

data/
├── raw/
│   ├── baseline_data/
│   │   ├── *.htm
│   │   └── *_files/
│   └── pdfs/
│       ├── 2025-operating-budget.pdf
│       ├── 9622_amusement_tax_regulations.pdf
│       ├── 9623_isp_tax_regulations.pdf
│       ├── 9624_local_services_tax_regulations.pdf
│       ├── 9625_parking_tax_regulations.pdf
│       └── 9626_payroll_tax_regulations.pdf
├── processed/
│   ├── scraped_websites.jsonl
│   └── chunks.jsonl
└── index/
    ├── dense.index
    ├── metadata.json
    └── bm25_index/
        ├── data.csc.index.npy
        ├── indices.csc.index.npy
        ├── indptr.csc.index.npy
        ├── params.index.json
        └── vocab.index.json

system_outputs/
├── submission_dense.json
├── submission_sparse.json
├── submission_hybrid.json
└── query_debug.jsonl

leaderboard_queries.json
```

## Setup & Installation
Run from repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Crucial Context (Apple Silicon M4 Pro)
- Development machine: Apple Silicon MacBook Pro (M4 Pro).
- Retrieval stack uses `faiss-cpu` on macOS (no CUDA/GPU flags required).
- Dynamic scraping uses Selenium and requires local Google Chrome; `webdriver-manager` resolves ChromeDriver.
- Generation uses local Ollama; pull model first:

```bash
ollama pull llama3.1
```

## Reproducible Pipeline Execution
All commands assume virtual environment is active.

### Option A: Full Build (Stages 1-4)
This clears `data/processed/` and `data/index/`, then rebuilds ingestion, chunking, dense index, and sparse index:

```bash
python -m src.pipeline
```

Optional: run generation right after build:

```bash
RUN_GENERATION_AFTER_BUILD=1 python -m src.pipeline
```

### Option B: Manual Run (Debug-Friendly)
#### Step 1: Data Ingestion
```bash
python -m src.ingestion.data_ingestion
```
Output:
- `data/processed/scraped_websites.jsonl`

#### Step 2: Chunking
```bash
python -m src.chunking.text_splitter
```
Output:
- `data/processed/chunks.jsonl`

Quick check:
```bash
wc -l data/processed/chunks.jsonl
```
Chunk count can vary due to dynamic event pages.

#### Step 3: Dense Index
```bash
python -m src.retrieval.dense
```
Outputs:
- `data/index/dense.index`
- `data/index/metadata.json`

#### Step 4: Sparse Index
```bash
python -m src.retrieval.sparse
```
Output:
- `data/index/bm25_index/*`

#### Step 5: End-to-End QA Generation
Set retrieval mode in `src/main.py`:
- `RETRIEVAL_MODE = "dense"` for FAISS-only
- `RETRIEVAL_MODE = "sparse"` for BM25-only
- `RETRIEVAL_MODE = "hybrid"` for fused retrieval

Then run:

```bash
python -m src.main
```

Inputs:
- `leaderboard_queries.json`
- `data/index/`

Outputs:
- `system_outputs/submission_dense.json` when mode is `dense`
- `system_outputs/submission_sparse.json` when mode is `sparse`
- `system_outputs/submission_hybrid.json` when mode is `hybrid`
- `system_outputs/query_debug.jsonl` per-query retrieval/answer log

## Architectural Notes
- URL targets are config-driven via `config/urls.json`.
- Ingestion is multi-source: baseline local HTML + static web requests + local PDFs + dynamic Selenium pages.
- Dynamic ingestion includes optional depth-1 same-domain link expansion for event-heavy pages.
- HTML cleaning preserves line structure (instead of fully flattening) to keep date/address/phone cues.
- Chunking uses a sliding window (`chunk_size=1000`, `overlap=200`) and emits chunk metadata (`source_type`, boundaries, length).
- Retrieval supports 3 hardcoded modes (`dense`, `sparse`, `hybrid`) in `src/retrieval/searcher.py`.
- Hybrid retrieval uses dense+BM25 candidate pools, RRF fusion, lightweight reranking, and per-document diversity cap.
- Reader uses local Ollama (`llama3.1`) and currently returns raw model text with `.strip()` only (no post-processing in `generate_answer`).

