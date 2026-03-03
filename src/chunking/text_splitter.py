import json
from pathlib import Path

# Tuning knobs for retrieval quality.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_overlapping_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append((start, end, chunk_text))
        if end >= len(text):
            break
    return chunks


def infer_source_type(doc_id: str) -> str:
    doc = doc_id.lower()
    if doc.endswith(".pdf"):
        return "pdf"
    if "wikipedia" in doc:
        return "wikipedia_html"
    if doc.startswith("http"):
        return "web"
    return "local_html"


if __name__ == "__main__":
    input_file = Path("data/processed/scraped_websites.jsonl")
    output_file = Path("data/processed/chunks.jsonl")

    all_chunks = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            url = doc["url"]
            text = doc["text"]
            source_type = infer_source_type(url)
            chunks = create_overlapping_chunks(text)

            for i, (start, end, chunk_text) in enumerate(chunks):
                all_chunks.append(
                    {
                        "doc_id": url,
                        "chunk_id": f"{url}_chunk_{i}",
                        "text": chunk_text,
                        "source_type": source_type,
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_len": len(chunk_text),
                    }
                )

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk_data in all_chunks:
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

    print(f"Split {len(all_chunks)} total chunks successfully!")
