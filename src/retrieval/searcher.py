import json
import re
import faiss
import bm25s
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATE_QUERY_HINTS = ("when", "what year", "what month", "what date", "dates")
NUMERIC_QUERY_HINTS = (
    "how many", "how much", "length", "capacity", "rate", "budget",
    "phone", "number", "acres", "miles", "feet", "blocks", "percent",
)
PHONE_HINTS = ("phone", "contact", "call")
ADDRESS_HINTS = ("address", "street", "located", "where")

PHONE_PATTERN = r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}"
DATE_PATTERN = (
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},\s+\d{4}\b"
)
YEAR_PATTERN = r"\b(16|17|18|19|20)\d{2}\b"

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "by", "from", "is",
    "are", "was", "were", "be", "being", "been", "what", "which", "who", "when",
    "where", "how", "many", "much", "does", "do", "did", "and", "or", "with",
    "that", "this", "it", "as", "into", "during", "about",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _query_flags(query: str) -> dict[str, bool]:
    q = query.lower()
    return {
        "date": any(h in q for h in DATE_QUERY_HINTS),
        "numeric": any(h in q for h in NUMERIC_QUERY_HINTS),
        "phone": any(h in q for h in PHONE_HINTS),
        "address": any(h in q for h in ADDRESS_HINTS),
    }


class HybridRetriever:
    def __init__(
        self,
        index_dir: Path,
        dense_candidate_k: int = 80,
        sparse_candidate_k: int = 80,
        rrf_c: int = 60,
        rerank_pool_k: int = 80,
        max_per_doc: int = 2,
        retrieval_mode: str = "hybrid",
    ):
        print("Loading indexes and models...")
        with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.dense_index = faiss.read_index(str(index_dir / "dense.index"))
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.sparse_index = bm25s.BM25.load(str(index_dir / "bm25_index"), load_corpus=False)

        self.dense_candidate_k = dense_candidate_k
        self.sparse_candidate_k = sparse_candidate_k
        self.rrf_c = rrf_c
        self.rerank_pool_k = rerank_pool_k
        self.max_per_doc = max_per_doc
        self.retrieval_mode = retrieval_mode.lower()
        if self.retrieval_mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("retrieval_mode must be one of: hybrid, dense, sparse")

        # Pre-tokenize chunks once for cheap lexical overlap reranking.
        self.chunk_tokens = [_tokenize(m.get("text", "")) for m in self.metadata]

    def search_dense(self, query: str, k: int = 10) -> list[int]:
        query_vec = self.encoder.encode([query]).astype("float32")
        _, indices = self.dense_index.search(query_vec, k)
        return [int(i) for i in indices[0].tolist()]

    def search_sparse(self, query: str, k: int = 10) -> list[int]:
        query_tokens = bm25s.tokenize([query])
        results, _ = self.sparse_index.retrieve(query_tokens, k=k)
        return [int(i) for i in results[0].tolist()]

    def _rrf_scores(self, dense_ranks: list[int], sparse_ranks: list[int]) -> dict[int, float]:
        scores = {}
        for rank, chunk_id in enumerate(dense_ranks):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_c + rank + 1)
        for rank, chunk_id in enumerate(sparse_ranks):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_c + rank + 1)
        return scores

    def _rerank(self, query: str, candidate_ids: list[int], base_scores: dict[int, float]) -> list[int]:
        q_tokens = {t for t in _tokenize(query) if t not in STOPWORDS}
        flags = _query_flags(query)
        max_base = max(base_scores.values()) if base_scores else 1.0

        ranked = []
        for chunk_id in candidate_ids:
            text = self.metadata[chunk_id].get("text", "")
            c_tokens = self.chunk_tokens[chunk_id]
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)

            signal_bonus = 0.0
            if flags["date"] and (re.search(DATE_PATTERN, text, flags=re.IGNORECASE) or re.search(YEAR_PATTERN, text)):
                signal_bonus += 0.15
            if flags["numeric"] and re.search(r"\d", text):
                signal_bonus += 0.10
            if flags["phone"] and re.search(PHONE_PATTERN, text):
                signal_bonus += 0.20
            if flags["address"] and re.search(r"\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr)\b", text, flags=re.IGNORECASE):
                signal_bonus += 0.12

            exact_phrase_bonus = 0.08 if query.lower() in text.lower() else 0.0
            rrf_norm = base_scores.get(chunk_id, 0.0) / max_base
            final_score = (0.62 * rrf_norm) + (0.30 * overlap) + signal_bonus + exact_phrase_bonus
            ranked.append((final_score, chunk_id))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [chunk_id for _, chunk_id in ranked]

    def _apply_diversity_cap(self, ranked_ids: list[int], k: int) -> list[int]:
        selected = []
        per_doc = {}
        for chunk_id in ranked_ids:
            doc_id = self.metadata[chunk_id].get("doc_id", "")
            used = per_doc.get(doc_id, 0)
            if used >= self.max_per_doc:
                continue
            selected.append(chunk_id)
            per_doc[doc_id] = used + 1
            if len(selected) >= k:
                break

        # If diversity cap is too strict, fill remaining slots without cap.
        if len(selected) < k:
            for chunk_id in ranked_ids:
                if chunk_id in selected:
                    continue
                selected.append(chunk_id)
                if len(selected) >= k:
                    break
        return selected

    def hybrid_search(self, query: str, k: int = 8) -> list[dict]:
        if self.retrieval_mode == "dense":
            dense_ids = self.search_dense(query, k=max(self.dense_candidate_k, k))
            final_ids = self._apply_diversity_cap(dense_ids, k=k)
            return [self.metadata[int(chunk_id)] for chunk_id in final_ids]

        if self.retrieval_mode == "sparse":
            sparse_ids = self.search_sparse(query, k=max(self.sparse_candidate_k, k))
            final_ids = self._apply_diversity_cap(sparse_ids, k=k)
            return [self.metadata[int(chunk_id)] for chunk_id in final_ids]

        dense_ids = self.search_dense(query, k=self.dense_candidate_k)
        sparse_ids = self.search_sparse(query, k=self.sparse_candidate_k)

        rrf_scores = self._rrf_scores(dense_ids, sparse_ids)
        candidate_ids = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
        candidate_ids = candidate_ids[: self.rerank_pool_k]

        reranked_ids = self._rerank(query, candidate_ids, rrf_scores)
        final_ids = self._apply_diversity_cap(reranked_ids, k=k)
        return [self.metadata[int(chunk_id)] for chunk_id in final_ids]


if __name__ == "__main__":
    index_folder = Path("data/index")
    retriever = HybridRetriever(index_folder)
    print(retriever.hybrid_search("When was Carnegie Mellon University founded?", k=5)[0]["doc_id"])
