import json
import time
import re
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pypdf import PdfReader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

EVENT_LINK_HINTS = (
    "event", "events", "calendar", "schedule", "show", "shows", "performance",
    "performances", "concert", "festival", "tickets", "2026",
)


def load_urls_config(config_path: str) -> dict:
    """Loads target URLs from a JSON configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Configuration file not found at {config_path}")
        return {"STATIC_URLS": [], "DYNAMIC_URLS": []}


def clean_html_to_text(html_content: str) -> str:
    """Extracts text while preserving useful line boundaries."""
    soup = BeautifulSoup(html_content, "html.parser")

    tags_to_decompose = ["script", "style", "nav", "footer", "aside", "form", "header"]
    for element in soup(tags_to_decompose):
        element.decompose()

    wiki_classes = ["reflist", "navbox", "reference", "noprint", "mw-editsection", "infobox"]
    for element in soup.find_all(["div", "span", "table"], class_=wiki_classes):
        element.decompose()

    raw_text = soup.get_text(separator="\n")
    raw_text = re.sub(r"\[\s*[a-zA-Z0-9]*\s*\]", "", raw_text)
    raw_text = re.sub(r'Retrieved from\s+"https?://\S+"', "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"WikiMiniAtlas", "", raw_text, flags=re.IGNORECASE)
    raw_text = raw_text.replace("\ufeff", "")

    normalized_lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        if len(line) == 1 and line in {"|", "•", "·"}:
            continue
        normalized_lines.append(line)

    clean_text = "\n".join(normalized_lines)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
    return clean_text


def process_local_baseline(directory_path: str) -> list[dict]:
    documents = []
    for file_path in Path(directory_path).rglob("*.htm*"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
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
        if extracted:
            full_text += extracted + "\n\n"
    full_text = re.sub(r"\r\n?", "\n", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    return full_text


def extract_child_links(html: str, parent_url: str, max_links: int = 3) -> list[str]:
    """Selects high-value same-domain links (depth=1) for event-heavy pages."""
    soup = BeautifulSoup(html, "html.parser")
    parent = urlparse(parent_url)
    selected = []
    seen = set()

    for tag in soup.find_all("a", href=True):
        href = (tag.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        abs_url = urljoin(parent_url, href)
        parsed = urlparse(abs_url)

        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != parent.netloc:
            continue

        path = parsed.path.lower()
        if not any(h in path for h in EVENT_LINK_HINTS):
            continue
        if any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".svg", ".css", ".js", ".xml", ".ico", ".pdf"]):
            continue

        normalized = parsed._replace(query="", fragment="").geturl()
        if normalized in seen or normalized == parent_url:
            continue
        seen.add(normalized)
        selected.append(normalized)

        if len(selected) >= max_links:
            break

    return selected


def scrape_dynamic_urls_with_selenium(
    urls: list[str],
    expand_depth1: bool = True,
    max_child_links: int = 3,
) -> list[dict]:
    print("Initializing Headless Chrome...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    documents = []
    visited = set()

    try:
        for url in urls:
            if url in visited:
                continue

            print(f"Dynamically scraping: {url}")
            try:
                driver.get(url)
                time.sleep(4)
                page_html = driver.page_source
                clean_text = clean_html_to_text(page_html)
                visited.add(url)

                if len(clean_text) > 50:
                    documents.append({"url": url, "text": clean_text})
                    print(f"Success! Extracted {len(clean_text)} characters.")
                else:
                    print(f"Warning: Extracted text is too short for {url}.")

                # Optional depth-1 crawl for event-heavy detail pages.
                if expand_depth1:
                    child_links = extract_child_links(page_html, url, max_links=max_child_links)
                    for child_url in child_links:
                        if child_url in visited:
                            continue
                        try:
                            print(f"  -> Following child page: {child_url}")
                            driver.get(child_url)
                            time.sleep(2.5)
                            child_text = clean_html_to_text(driver.page_source)
                            visited.add(child_url)
                            if len(child_text) > 50:
                                documents.append({"url": child_url, "text": child_text})
                        except Exception as child_err:
                            print(f"Warning: Failed child page {child_url}: {child_err}")

            except Exception as e:
                print(f"Error scraping dynamic URL {url}: {e}")
    finally:
        driver.quit()
    return documents


def save_documents(docs: list[dict], output_path: Path, mode: str = "a"):
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

    output_file = Path("data/processed/scraped_websites.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Processing Baseline Data ---")
    baseline_docs = process_local_baseline("data/raw/baseline_data")
    save_documents(baseline_docs, output_file, mode="w")
    print(f"Saved {len(baseline_docs)} baseline documents.")

    print("\n--- Processing Static URLs ---")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
        )
    }
    static_docs = []
    for url in STATIC_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=12)
            if response.status_code == 200:
                clean_text = clean_html_to_text(response.text)
                if len(clean_text) > 50:
                    static_docs.append({"url": url, "text": clean_text})
                    print(f"Successfully scraped: {url}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    save_documents(static_docs, output_file)

    print("\n--- Processing Local PDFs ---")
    pdf_dir = Path("data/raw/pdfs")
    pdf_docs = []
    if pdf_dir.exists():
        for pdf_path in pdf_dir.glob("*.pdf"):
            pdf_text = parse_local_pdf(str(pdf_path))
            if len(pdf_text) > 50:
                pdf_docs.append({"url": pdf_path.name, "text": pdf_text})
                print(f"Successfully parsed: {pdf_path.name}")
        if pdf_docs:
            save_documents(pdf_docs, output_file)
            print(f"Saved {len(pdf_docs)} total PDF documents.")
    else:
        print(f"WARNING: Directory {pdf_dir} not found. Skipping PDFs.")

    print("\n--- Processing Dynamic URLs ---")
    dynamic_docs = scrape_dynamic_urls_with_selenium(DYNAMIC_URLS, expand_depth1=True, max_child_links=3)
    save_documents(dynamic_docs, output_file)

    print(f"\nCOMPLETE! All data unified in {output_file}")
