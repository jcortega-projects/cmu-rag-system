import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_step(module_name: str) -> None:
    """Executes a Python module as a separate process."""
    print(f"Starting step: {module_name}")
    start_time = time.time()
    try:
        subprocess.check_call([sys.executable, "-m", module_name])
        elapsed_time = time.time() - start_time
        print(f"Completed step: {module_name} in {elapsed_time:.2f} seconds.\n")
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: Pipeline halted. {module_name} failed with exit code {e.returncode}.")
        sys.exit(1)


def clean_environment() -> None:
    print("Cleaning previous build artifacts...")
    processed_dir = Path("data/processed")
    index_dir = Path("data/index")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    print("Clean slate ready.\n")


def main() -> None:
    print("Initializing RAG Build Pipeline...\n")
    pipeline_start = time.time()

    # Default build: ingestion -> chunking -> dense index -> sparse index
    steps = [
        "src.ingestion.data_ingestion",
        "src.chunking.text_splitter",
        "src.retrieval.dense",
        "src.retrieval.sparse",
    ]

    # Optional generation pass: RUN_GENERATION_AFTER_BUILD=1 python -m src.pipeline
    if os.getenv("RUN_GENERATION_AFTER_BUILD", "0") == "1":
        steps.append("src.main")

    clean_environment()
    for step in steps:
        run_step(step)

    pipeline_elapsed = time.time() - pipeline_start
    print("--------------------------------------------------")
    print(f"Pipeline execution finished successfully in {pipeline_elapsed:.2f} seconds.")
    print("Indexes are up to date. Generation stage ran only if explicitly enabled.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
