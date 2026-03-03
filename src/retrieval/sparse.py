import json
import bm25s
from pathlib import Path

def build_sparse_index(chunks_path: Path, index_dir: Path):
    print("1. Loading chunk data for BM25...")
    texts = []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            # We only need the text for BM25 to build its vocabulary
            texts.append(doc['text'])
            
    print(f"Loaded {len(texts)} chunks.")
    
    print("2. Tokenizing texts...")
    # bm25s has a highly optimized C++ tokenizer under the hood
    corpus_tokens = bm25s.tokenize(texts)
    
    print("3. Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    print("4. Saving BM25 index to disk...")
    # bm25s saves multiple files (vocab, scores, etc.), so give it a dedicated subfolder
    bm25_dir = index_dir / "bm25_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    
    retriever.save(str(bm25_dir))
    
    print(f"Sparse index saved to {bm25_dir}/")

if __name__ == "__main__":
    chunks_file = Path("data/processed/chunks.jsonl")
    output_folder = Path("data/index")
    
    build_sparse_index(chunks_file, output_folder)