import json
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer
from pathlib import Path

def build_dense_index(chunks_path: Path, index_dir: Path):
    print("1. Loading chunk data...")
    texts = []
    metadata = []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc['text'])
            # Save the metadata so we can look up the text later
            metadata.append({
                "doc_id": doc["doc_id"], 
                "chunk_id": doc["chunk_id"],
                "text": doc["text"] # We store the text here for easy retrieval
            })
            
    print(f"Loaded {len(texts)} chunks.")
    
    print("2. Loading embedding model (all-MiniLM-L6-v2)...")
    # This downloads the model on the first run, then caches it locally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("3. Encoding chunks into vectors...")
    # The encode function automatically handles tokenization and batching
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # FAISS strictly requires float32 data types
    embeddings = np.array(embeddings).astype('float32')
    
    print("4. Building FAISS index...")
    dimension = embeddings.shape[1] # This will be 384
    
    # IndexFlatL2 measures Euclidean distance between vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Index built with {index.ntotal} vectors.")
    
    # 5. Save the index and metadata to disk
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "dense.index"))
    
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
        
    print(f"Dense index and metadata saved to {index_dir}/")

if __name__ == "__main__":
    chunks_file = Path("data/processed/chunks.jsonl")
    output_folder = Path("data/index")
    
    build_dense_index(chunks_file, output_folder)