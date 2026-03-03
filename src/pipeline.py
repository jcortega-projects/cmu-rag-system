import subprocess
import sys
import time
from pathlib import Path
import shutil

def run_step(module_name: str) -> None:
    """Executes a Python module as a separate process."""
    print(f"Starting step: {module_name}")
    start_time = time.time()
    
    try:
        # sys.executable ensures we use the active virtual environment (e.g., rag_env)
        # We use the '-m' flag to run it as a module, preserving your relative imports
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
    
    # Delete the folders and everything inside them
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
        
    # Recreate the empty folders
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    print("Clean slate ready.\n")

def main() -> None:
    print("Initializing RAG Build Pipeline...\n")
    pipeline_start = time.time()
    
    # The chronological order is strictly enforced here
    steps = [
        "src.ingestion.data_ingestion",
        "src.chunking.text_splitter",
        "src.retrieval.dense",
        "src.retrieval.sparse"
    ]

    clean_environment()
    for step in steps:
        run_step(step)
        
    pipeline_elapsed = time.time() - pipeline_start
    print("--------------------------------------------------")
    print(f"Pipeline execution finished successfully in {pipeline_elapsed:.2f} seconds.")
    print("Your vector and keyword indexes are fully up to date.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()