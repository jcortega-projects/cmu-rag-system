import json
import numpy as np
import faiss
import bm25s
from sentence_transformers import SentenceTransformer
from pathlib import Path

class HybridRetriever:
    def __init__(self, index_dir: Path):
        print("Loading indexes and models...")
        # 1. Load the "Coat Check Ledger" (Metadata)
        with open(index_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        # 2. Load Dense Index (FAISS) & Model
        self.dense_index = faiss.read_index(str(index_dir / "dense.index"))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Load Sparse Index (BM25)
        self.sparse_index = bm25s.BM25.load(str(index_dir / "bm25_index"), load_corpus=False)
        
    def search_dense(self, query: str, k: int = 5) -> list[int]:
        """Returns the top k integer IDs from FAISS."""
        # Convert query to vector
        query_vec = self.encoder.encode([query]).astype('float32')
        # FAISS search returns distances (D) and integer IDs (I)
        distances, indices = self.dense_index.search(query_vec, k)
        return indices[0].tolist()
        
    def search_sparse(self, query: str, k: int = 5) -> list[int]:
        """Returns the top k integer IDs from BM25."""
        query_tokens = bm25s.tokenize([query])
        # BM25 returns results (integer IDs) and scores
        results, scores = self.sparse_index.retrieve(query_tokens, k=k)
        return results[0].tolist()

    def reciprocal_rank_fusion(self, dense_ranks: list[int], sparse_ranks: list[int], k: int = 5) -> list[int]:
        """Fuses two ranked lists using RRF."""
        rrf_scores = {}
        c = 60 

        for rank, chunk_id in enumerate(dense_ranks):
            rrf_scores[chunk_id] = 1.0 / (c+rank+1)
       
        for rank, chunk_id in enumerate(sparse_ranks):
            rrf_score = 1.0 / (c+rank+1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score
      
        sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        final_chunk_ids = [item[0] for item in sorted_items][:k]
    
        return final_chunk_ids

    def hybrid_search(self, query: str, k: int = 5) -> list[dict]:
        """The main search function!"""
       
        dense_ids = self.search_dense(query, k=10)
        sparse_ids = self.search_sparse(query, k=10)
        final_ids = self.reciprocal_rank_fusion(dense_ids, sparse_ids, k=k)
        results = []
        for rank, chunk_id in enumerate(final_ids):
            chunk_data = self.metadata[chunk_id]
            results.append(chunk_data)
            # print(f"[{rank+1}] {chunk_data['doc_id']} (ID: {chunk_id})")
            
        return results

if __name__ == "__main__":
    index_folder = Path("data/index")
    retriever = HybridRetriever(index_folder)

    retriever.hybrid_search("When was Carnegie Mellon University founded?", k=3)
    retriever.hybrid_search("What is the penalty percentage for late Amusement Tax?", k=3)