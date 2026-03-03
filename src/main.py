import json
from pathlib import Path
from src.retrieval.searcher import HybridRetriever
from src.generation.reader import generate_answer


# ---------------------------------------
# CHANGE THIS TO SWITCH MODES
# ---------------------------------------
# Options:
#   "LEADERBOARD"
#   "UNSEEN_DAY5"
RUN_MODE = "UNSEEN_DAY5"
# ---------------------------------------

# Retrieval/generation knobs.
# hardcode retrieval mode here: "hybrid", "dense" (FAISS-only), "sparse" (BM25-only)
RETRIEVAL_MODE = "hybrid"
TOP_K = 10
DENSE_CANDIDATE_K = 100
SPARSE_CANDIDATE_K = 100
RRF_C = 55
RERANK_POOL_K = 120
MAX_PER_DOC = 2

WRITE_QUERY_DEBUG_LOG = True
QUERY_DEBUG_LOG_PATH = Path("system_outputs/query_debug.jsonl")


def load_leaderboard_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["id"], item["question"]) for item in data]


def load_unseen_txt(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [(str(i), q) for i, q in enumerate(lines, start=1)]


def run_end_to_end_evaluation(
    queries,
    output_path: Path,
    index_dir: Path,
    include_andrewid: bool,
    andrew_id: str,
    top_k: int = TOP_K,
):
    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever(
        index_dir=index_dir,
        dense_candidate_k=DENSE_CANDIDATE_K,
        sparse_candidate_k=SPARSE_CANDIDATE_K,
        rrf_c=RRF_C,
        rerank_pool_k=RERANK_POOL_K,
        max_per_doc=MAX_PER_DOC,
        retrieval_mode=RETRIEVAL_MODE,
    )
    print(f"Retrieval mode: {RETRIEVAL_MODE}")

    results = {}
    if include_andrewid:
        results["andrewid"] = andrew_id

    if WRITE_QUERY_DEBUG_LOG:
        QUERY_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUERY_DEBUG_LOG_PATH.write_text("", encoding="utf-8")

    print(f"Processing {len(queries)} queries...\n")

    for qid, query_text in queries:
        print(f"--- Question {qid}/{len(queries)} ---")
        print(f"Q: {query_text}")

        retrieved_chunks = retriever.hybrid_search(query_text, k=top_k)
        answer = generate_answer(query_text, retrieved_chunks)
        print(f"A: {answer}\n")

        results[str(qid)] = answer

        if WRITE_QUERY_DEBUG_LOG:
            debug_row = {
                "qid": str(qid),
                "query": query_text,
                "answer": answer,
                "retrieved_doc_ids": [c.get("doc_id") for c in retrieved_chunks],
                "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
            }
            with open(QUERY_DEBUG_LOG_PATH, "a", encoding="utf-8") as dbg:
                dbg.write(json.dumps(debug_row, ensure_ascii=False) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    index_folder = Path("data/index")

    if RUN_MODE == "LEADERBOARD":
        queries_file = Path("leaderboard_queries.json")
        queries = load_leaderboard_json(queries_file)
        output_file = Path(f"system_outputs/submission_{RETRIEVAL_MODE}.json")
        run_end_to_end_evaluation(
            queries=queries,
            output_path=output_file,
            index_dir=index_folder,
            include_andrewid=True,
            andrew_id="jcortega",
            top_k=TOP_K,
        )

    elif RUN_MODE == "UNSEEN_DAY5":
        queries_file = Path("test_set_day_5.txt")
        queries = load_unseen_txt(queries_file)
        output_file = Path(f"system_outputs/system_output_1.json")
        run_end_to_end_evaluation(
            queries=queries,
            output_path=output_file,
            index_dir=index_folder,
            include_andrewid=False,
            andrew_id="jcortega",
            top_k=TOP_K,
        )
