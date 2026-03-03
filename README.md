# CMU Pittsburgh-Centric RAG Pipeline

## Project Overview
This repository contains a custom Retrieval-Augmented Generation (RAG) system built for CMU Advanced NLP to answer hyper-local, temporal questions about Pittsburgh and Carnegie Mellon University. The current pipeline is fully modular and includes ingestion, chunking, dense + sparse indexing, hybrid retrieval, and local LLM answer generation for leaderboard submission.

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
└── submission.json

leaderboard_queries.txt
```

## Setup & Installation
Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Crucial Context (Apple Silicon M4 Pro)
- Development environment: Apple Silicon MacBook Pro (M4 Pro).
- Retrieval stack uses `faiss-cpu` (no CUDA/GPU flags required on macOS).
- Selenium dynamic scraping requires local Google Chrome; `webdriver-manager` resolves ChromeDriver automatically.
- Generation uses local Ollama from Python (`ollama` package). Install Ollama app and pull model once:

```bash
ollama pull llama3.1
```

## Reproducible Pipeline Execution
All commands below assume you are at repo root with the virtual environment activated.

### Option A: Run Full Build (Stages 1-4) with One Command
This clears `data/processed/` and `data/index/` first, then rebuilds ingestion, chunking, dense index, and sparse index:

```bash
python -m src.pipeline
```

### Option B: Run Stages Manually (Recommended for Debugging)
#### Step 1: Data Ingestion
```bash
python -m src.ingestion.data_ingestion
```
Output:
- `data/processed/scraped_websites.jsonl`

#### Step 2: Text Chunking
```bash
python -m src.chunking.text_splitter
```
Output:
- `data/processed/chunks.jsonl`

Sanity check:
```bash
wc -l data/processed/chunks.jsonl
```
Current checked output in this workspace is `3171`, but this can vary because dynamic event/news pages change over time.

#### Step 3: Dense Vector Indexing (FAISS)
```bash
python -m src.retrieval.dense
```
Outputs:
- `data/index/dense.index`
- `data/index/metadata.json`

#### Step 4: Sparse Indexing (BM25)
```bash
python -m src.retrieval.sparse
```
Output:
- `data/index/bm25_index/*`

#### Step 5: Generation / End-to-End Leaderboard File
```bash
python -m src.main
```
Inputs:
- `leaderboard_queries.txt`
- `data/index/` (dense + sparse + metadata)

Output:
- `system_outputs/submission.json`

## Additional Run Commands
Test hybrid retrieval only:

```bash
python -m src.retrieval.searcher
```

## Architectural Notes
- URL targets are config-driven (`config/urls.json`) so static and dynamic source lists can be expanded without changing ingestion code.
- Ingestion is multi-modal and unified: baseline local HTML snapshots + static pages (`requests`) + JS-heavy dynamic pages (Selenium headless Chrome) + dynamic PDF directory loading from `data/raw/pdfs/*.pdf`.
- HTML normalization includes generic boilerplate stripping and Wikipedia-specific noise removal (`reflist`, `navbox`, `reference`, `mw-editsection`, `infobox`) to reduce retrieval contamination.
- Chunking uses a sliding window with `chunk_size=1000` and `overlap=200` to preserve semantic continuity around boundary spans.
- Dense retrieval uses `all-MiniLM-L6-v2` embeddings with FAISS `IndexFlatL2`.
- Sparse retrieval uses BM25 via `bm25s` and stores a serialized sub-index under `data/index/bm25_index/`.
- Hybrid retrieval (`src/retrieval/searcher.py`) fuses dense and sparse rankings using Reciprocal Rank Fusion (RRF), then returns top-k chunks for answer generation.
- Reader/generator (`src/generation/reader.py`) calls local `llama3.1` through Ollama with a strict prompt that enforces concise answers and `"Not found"` when evidence is absent.

## Notes for TAs
- If `src.main` fails at generation time, verify Ollama is installed/running and `llama3.1` is available locally.
- `src.main` currently hardcodes `andrew_id="jcortega"` in code. Update this field in `src/main.py` if you are running under a different identifier.
