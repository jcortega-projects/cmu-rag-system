import json
from pathlib import Path
from src.ingestion.data_ingestion import parse_local_pdf, save_documents

def patch_missing_pdfs():
    pdf_dir = Path("data/raw/pdfs")
    output_file = Path("data/processed/scraped_websites.jsonl")
    
    new_docs = []
    
    # Loop through all PDFs you just downloaded
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_path.name}...")
        pdf_text = parse_local_pdf(str(pdf_path))
        
        if len(pdf_text) > 50:
            new_docs.append({
                "url": pdf_path.name,
                "text": pdf_text
            })
            
    # Append them to the master file
    if new_docs:
        save_documents(new_docs, output_file, mode="a")
        print(f"✅ Successfully appended {len(new_docs)} tax regulations to the master file!")
    else:
        print("⚠️ No PDFs found or parsed.")

if __name__ == "__main__":
    patch_missing_pdfs()