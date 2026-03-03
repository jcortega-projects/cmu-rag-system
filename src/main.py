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
RUN_MODE = "LEADERBOARD"
# ---------------------------------------


def load_leaderboard_json(path: Path):
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # returns list of (id, question)
    return [(item["id"], item["question"]) for item in data]


def load_unseen_txt(path: Path):
    
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # enumerate to create ids "1", "2", ...
    return [(str(i), q) for i, q in enumerate(lines, start=1)]


def run_end_to_end_evaluation(
    queries,
    output_path: Path,
    index_dir: Path,
    include_andrewid: bool,
    andrew_id: str,
    top_k=3
):

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever(index_dir)

    results = {}

    # Leaderboard requires this field
    if include_andrewid:
        results["andrewid"] = andrew_id

    print(f"Processing {len(queries)} queries...\n")

    # DO NOT DELETE (your previous variants)
    #for idx, query_text in enumerate(queries, start=1):
    #for idx, query_text in queries.items():

    for qid, query_text in queries:
        print(f"--- Question {qid}/{len(queries)} ---")
        print(f"Q: {query_text}")

        retrieved_chunks = retriever.hybrid_search(query_text, k=top_k)
        answer = generate_answer(query_text, retrieved_chunks)

        print(f"A: {answer}\n")

        results[str(qid)] = answer

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":

    index_folder = Path("data/index")

    if RUN_MODE == "LEADERBOARD":
        queries_file = Path("leaderboard_queries.json")
        queries = load_leaderboard_json(queries_file)

        output_file = Path("system_outputs/submission.json")

        run_end_to_end_evaluation(
            queries=queries,
            output_path=output_file,
            index_dir=index_folder,
            include_andrewid=True, 
            andrew_id="jcortega",
            top_k=5
        )

    elif RUN_MODE == "UNSEEN_DAY5":
        queries_file = Path("test_set_day_5.txt")
        queries = load_unseen_txt(queries_file)

        output_file = Path("system_outputs/system_output_1.json")

        run_end_to_end_evaluation(
            queries=queries,
            output_path=output_file,
            index_dir=index_folder,
            include_andrewid=False,  
            andrew_id="jcortega",
            top_k=5
        )