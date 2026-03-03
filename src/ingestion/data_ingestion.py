import json
import time
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def load_urls_config(config_path: str) -> dict:
    """Loads target URLs from a JSON configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Configuration file not found at {config_path}")
        return {"STATIC_URLS": [], "DYNAMIC_URLS": []}

def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Standard HTML tags to destroy
    tags_to_decompose = ['script', 'style', 'nav', 'footer', 'aside', 'form', 'header']
    for element in soup(tags_to_decompose):
        element.decompose()
        
    # 2. Wikipedia-specific noise to destroy (References, Navigation boxes, Edit buttons)
    wiki_classes = ['reflist', 'navbox', 'reference', 'noprint', 'mw-editsection', 'infobox']
    for element in soup.find_all(['div', 'span', 'table'], class_=wiki_classes):
        element.decompose()

    # 3. Extract text, replacing block elements with a space instead of newlines
    clean_text = soup.get_text(separator=" ") 
    
   # 4. Regex Normalization
    # Broad catch for any bracketed text with numbers or short words (e.g., [1], [ 28 ], [edit], [a])
    clean_text = re.sub(r'\[\s*[a-zA-Z0-9]*\s*\]', '', clean_text)
    
    # Remove Wikipedia's hidden "Retrieved from" URLs and WikiMiniAtlas artifacts
    clean_text = re.sub(r'Retrieved from\s+"https?://\S+"', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'WikiMiniAtlas', '', clean_text, flags=re.IGNORECASE)
    
    # Remove zero-width unicode spaces
    clean_text = clean_text.replace('\ufeff', '')
    
    # The Nuke: Squash ANY combination of spaces, tabs, and newlines (\n) into a single space
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()

def process_local_baseline(directory_path: str) -> list[dict]:
    documents = []
    for file_path in Path(directory_path).rglob('*.htm*'): 
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            clean_text = clean_html_to_text(f.read())
            if len(clean_text) > 50: 
                documents.append({"url": file_path.name, "text": clean_text})
    return documents

def parse_local_pdf(file_path: str) -> str:
    print(f"Parsing local PDF: {file_path}")
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted: full_text += extracted + "\n\n"
    return re.sub(r'\n{3,}', '\n\n', full_text).strip()

def scrape_dynamic_urls_with_selenium(urls: list[str]) -> list[dict]:
    print("Initializing Headless Chrome...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    documents = []
    
    try:
        for url in urls:
            print(f"Dynamically scraping: {url}")
            driver.get(url)
            time.sleep(4) 
            clean_text = clean_html_to_text(driver.page_source)
            if len(clean_text) > 50:
                documents.append({"url": url, "text": clean_text})
                print(f"Success! Extracted {len(clean_text)} characters.")
            else:
                print(f"Warning: Extracted text is too short for {url}.")
    finally:
        driver.quit()
    return documents

def save_documents(docs: list[dict], output_path: Path, mode: str = "a"):
    """Helper function to save documents to JSONL."""
    with open(output_path, mode, encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    print("\n--- Loading Configuration ---")
    config_path = Path("config/urls.json")
    urls_config = load_urls_config(str(config_path))
    STATIC_URLS = urls_config.get("STATIC_URLS", [])
    DYNAMIC_URLS = urls_config.get("DYNAMIC_URLS", [])
    print(f"Loaded {len(STATIC_URLS)} static URLs and {len(DYNAMIC_URLS)} dynamic URLs.")
    
    # 0. Setup Output File
    output_file = Path("data/processed/scraped_websites.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Process Baseline Data (Use "w" to overwrite old runs and start fresh)
    print("\n--- Processing Baseline Data ---")
    baseline_docs = process_local_baseline("data/raw/baseline_data")
    save_documents(baseline_docs, output_file, mode="w")
    print(f"Saved {len(baseline_docs)} baseline documents.")
    
    # 2. Static Web Pages
    print("\n--- Processing Static URLs ---")
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36'}
    static_docs = []
    for url in STATIC_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                clean_text = clean_html_to_text(response.text)
                if len(clean_text) > 50: 
                    static_docs.append({"url": url, "text": clean_text})
                    print(f"Successfully scraped: {url}")
        except Exception as e: print(f"Error scraping {url}: {e}")
    save_documents(static_docs, output_file)
    
    # 3. PDF Parsing
    print("\n--- Processing Local PDFs ---")
    # Point this to the folder where ALL your PDFs live
    pdf_dir = Path("data/raw/pdfs") 
    pdf_docs = []

    if pdf_dir.exists():
        # Iterate through every PDF in the directory dynamically
        for pdf_path in pdf_dir.glob("*.pdf"):
            pdf_text = parse_local_pdf(str(pdf_path))
            if len(pdf_text) > 50:
                pdf_docs.append({"url": pdf_path.name, "text": pdf_text})
                print(f"Successfully parsed: {pdf_path.name}")
        
        # Save them all at once
        if pdf_docs:
            save_documents(pdf_docs, output_file)
            print(f"Saved {len(pdf_docs)} total PDF documents.")
    else:
        print(f"WARNING: Directory {pdf_dir} not found. Skipping PDFs.")
    
    # 4. Dynamic Web Pages
    print("\n--- Processing Dynamic URLs ---")
    dynamic_docs = scrape_dynamic_urls_with_selenium(DYNAMIC_URLS)
    save_documents(dynamic_docs, output_file)
    
    print(f"\nCOMPLETE! All data unified in {output_file}")