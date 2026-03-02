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

STATIC_URLS = [
    "https://en.wikipedia.org/wiki/Pittsburgh",
    "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "https://www.britannica.com/place/Pittsburgh",
    "https://www.cmu.edu/about/",
]

DYNAMIC_URLS = [
    "https://www.visitpittsburgh.com/",
    "https://events.cmu.edu/",
    "https://www.pghcitypaper.com/pittsburgh/EventSearch",
    "https://www.steelers.com/",
    "https://www.nhl.com/penguins/",
    "https://www.picklesburgh.com/",
]

def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'form', 'header']):
        element.decompose()
    clean_text = soup.get_text(separator="\n") 
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
    return clean_text

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
            f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
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
    pdf_path = "data/raw/2025-operating-budget.pdf"
    if Path(pdf_path).exists():
        pdf_text = parse_local_pdf(pdf_path)
        if len(pdf_text) > 50:
            save_documents([{"url": "2025-operating-budget.pdf", "text": pdf_text}], output_file)
            print("Successfully parsed and saved the PDF.")
    
    # 4. Dynamic Web Pages
    print("\n--- Processing Dynamic URLs ---")
    dynamic_docs = scrape_dynamic_urls_with_selenium(DYNAMIC_URLS)
    save_documents(dynamic_docs, output_file)
    
    print(f"\nCOMPLETE! All data unified in {output_file}")