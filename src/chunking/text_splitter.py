import json
from pathlib import Path

def create_overlapping_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
   
    chunks = []
    step = chunk_size - overlap
    
    # Sliding window approach
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break 
    return chunks

if __name__ == "__main__":
    input_file = Path("data/processed/scraped_websites.jsonl")
    output_file = Path("data/processed/chunks.jsonl")
    
    all_chunks = []
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            url = doc["url"]
            text = doc["text"]
            chunks = create_overlapping_chunks(text)

            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "doc_id": url,
                    "chunk_id": f"{url}_chunk_{i}",
                    "text": chunk_text
                })

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk_data in all_chunks:
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
            
    print(f"Split {len(all_chunks)} total chunks successfully!")