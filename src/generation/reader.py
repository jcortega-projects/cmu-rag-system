import re

NOT_FOUND = "Not Found"

DATE_QUERY_HINTS = (
    "when",
    "what year",
    "what month",
    "what date",
    "on what date",
    "on what dates",
    "dates",
)

NUMERIC_QUERY_HINTS = (
    "how many",
    "how much",
    "length",
    "capacity",
    "rate",
    "budget",
    "phone",
    "number",
    "acres",
    "miles",
    "feet",
    "blocks",
    "age",
)

PHONE_QUERY_HINTS = ("phone", "contact", "call")

ADDRESS_QUERY_HINTS = ("address", "street", "located", "where")

MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)

DATE_PATTERNS = [
    rf"\b{MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b",
    rf"\b{MONTH_PATTERN}\s+\d{{4}}\b",
    r"\b(16|17|18|19|20)\d{2}\b",
]

PHONE_PATTERN = r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}"
ADDRESS_PATTERN = (
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]{2,80}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|"
    r"Lane|Ln|Way|Court|Ct|Place|Pl)\b"
)

NUMBER_WITH_UNIT_PATTERN = (
    r"\$?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*%|\s*(?:acres?|miles?|feet|blocks?|sq\.?\s*miles?))?"
)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "by", "from", "is",
    "are", "was", "were", "be", "being", "been", "what", "which", "who", "when",
    "where", "how", "many", "much", "does", "do", "did", "and", "or", "with",
    "that", "this", "it", "as", "into", "during", "about", "name",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _query_flags(query: str) -> dict[str, bool]:
    q = query.lower()
    is_date_query = any(h in q for h in DATE_QUERY_HINTS)
    is_numeric_query = any(h in q for h in NUMERIC_QUERY_HINTS)
    return {
        "is_date_query": is_date_query,
        "is_numeric_query": is_numeric_query,
        "is_phone_query": any(h in q for h in PHONE_QUERY_HINTS),
        "is_address_query": any(h in q for h in ADDRESS_QUERY_HINTS),
        "is_entity_query": q.startswith(("who", "which", "what")) and not is_date_query and not is_numeric_query,
    }


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    import ollama

    context_text = "\n\n---\n\n".join(
        [f"Source: {chunk['doc_id']}\nText: {chunk['text']}" for chunk in retrieved_chunks]
    )

    system_prompt = f"""You are a highly precise, strict extractive QA bot. 

RULES:
1. Answer the question using ONLY the provided context.
2. Output ONLY the exact entity, date, number, or short phrase required. 
3. DO NOT write full sentences. DO NOT write conversational filler like "According to the context".
4. If the context does not contain the answer, output exactly: Not found

--- EXAMPLE 1 ---
Context: The Cathedral of Learning is a 42-story skyscraper that serves as the centerpiece of the University of Pittsburgh's main campus in the Oakland neighborhood.
Question: How tall is the Cathedral of Learning?
Answer: 42 stories

--- EXAMPLE 2 ---
Context: The Pittsburgh Steelers are a professional American football team.
Question: Who is the head coach of the Pittsburgh Steelers?
Answer: Not found

--- EXAMPLE 3 ---
Context: Andrew Mellon and Richard B. Mellon founded the institute in 1913.
Question: Who founded the institute?
Answer: Andrew Mellon and Richard B. Mellon

--- REAL TASK ---
Context:
{context_text}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": system_prompt}],
    )
    # Return raw model output (only stripped), without regex post-processing/fallback.
    raw_answer = response["message"]["content"]
    return raw_answer.strip()


def normalize_answer(query: str, answer: str) -> str:
    q_flags = _query_flags(query)
    a = (answer or "").strip()

    if not a:
        return NOT_FOUND

    # Keep first non-empty line.
    lines = [line.strip() for line in a.splitlines() if line.strip()]
    a = lines[0] if lines else ""
    if not a:
        return NOT_FOUND

    # Strip common wrapper prefixes.
    a = re.sub(r"^(answer|final answer)\s*:\s*", "", a, flags=re.IGNORECASE).strip()

    not_found_patterns = [
        r"\bnot found\b",
        r"\bnot explicitly (mentioned|stated|provided)\b",
        r"\bnot provided\b",
        r"\bcannot (be )?(determined|found)\b",
        r"\binsufficient (context|information)\b",
        r"\bthe (context|text) does(?:n'?t| not) (mention|provide)\b",
        r"\bi don'?t know\b",
    ]
    if any(re.search(pat, a, flags=re.IGNORECASE) for pat in not_found_patterns):
        return NOT_FOUND

    # Trim common explanation tails.
    a = re.split(r"\b(?:however|although|though|because)\b", a, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Question-aware span extraction on model output itself.
    if q_flags["is_phone_query"]:
        m = re.search(PHONE_PATTERN, a)
        if m:
            a = m.group(0)
    elif q_flags["is_address_query"]:
        m = re.search(ADDRESS_PATTERN, a, flags=re.IGNORECASE)
        if m:
            a = m.group(0)
    elif q_flags["is_date_query"]:
        for pat in DATE_PATTERNS:
            m = re.search(pat, a, flags=re.IGNORECASE)
            if m:
                a = m.group(0)
                break
    elif q_flags["is_numeric_query"]:
        m = re.search(NUMBER_WITH_UNIT_PATTERN, a, flags=re.IGNORECASE)
        if m:
            a = m.group(0)

    # Normalize punctuation noise.
    a = a.strip().strip("\"'` ")
    a = re.sub(r"(?<=\d),$", "", a)
    a = re.sub(r"[.;:]+$", "", a).strip()
    a = re.sub(r"\s+", " ", a).strip()

    return a if a else NOT_FOUND


def _should_use_fallback(query: str, answer: str) -> bool:
    if answer == NOT_FOUND:
        return True

    flags = _query_flags(query)
    lower = answer.lower()

    # Generic/meta responses tend to score poorly.
    if any(phrase in lower for phrase in ["according to", "in the context", "the text"]):
        return True

    # Badly-formed short numeric/date outputs (e.g., "18,").
    if re.fullmatch(r"\d{1,2},?", answer):
        return True

    if flags["is_date_query"]:
        has_date = any(re.search(p, answer, flags=re.IGNORECASE) for p in DATE_PATTERNS)
        if not has_date:
            return True

    if flags["is_phone_query"] and not re.search(PHONE_PATTERN, answer):
        return True

    if flags["is_address_query"] and not re.search(ADDRESS_PATTERN, answer, flags=re.IGNORECASE):
        return True

    if flags["is_numeric_query"] and not re.search(r"\d", answer):
        return True

    return False


def extract_from_context(query: str, retrieved_chunks: list[dict]) -> str:
    flags = _query_flags(query)
    query_tokens = [t for t in _tokenize(query) if t not in STOPWORDS]

    candidates = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        # Keep punctuation boundaries for better extraction.
        sentences = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
        for sent in sentences:
            s = sent.strip()
            if len(s) < 3:
                continue
            s_tokens = set(_tokenize(s))
            overlap = len(set(query_tokens) & s_tokens)
            if overlap > 0:
                candidates.append((overlap, s))

    if not candidates:
        return NOT_FOUND

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in candidates[:80]]

    # Rule-based extraction by query type.
    for sentence in top_sentences:
        if flags["is_phone_query"]:
            m = re.search(PHONE_PATTERN, sentence)
            if m:
                return normalize_answer(query, m.group(0))

        if flags["is_address_query"]:
            m = re.search(ADDRESS_PATTERN, sentence, flags=re.IGNORECASE)
            if m:
                return normalize_answer(query, m.group(0))

        if flags["is_date_query"]:
            for pat in DATE_PATTERNS:
                m = re.search(pat, sentence, flags=re.IGNORECASE)
                if m:
                    return normalize_answer(query, m.group(0))

        if flags["is_numeric_query"]:
            m = re.search(NUMBER_WITH_UNIT_PATTERN, sentence, flags=re.IGNORECASE)
            if m:
                return normalize_answer(query, m.group(0))

    # Conservative entity fallback.
    if flags["is_entity_query"]:
        for sentence in top_sentences[:12]:
            quoted = re.search(r'"([^"]{2,100})"', sentence)
            if quoted:
                candidate = quoted.group(1).strip()
                if candidate and len(candidate.split()) <= 10:
                    return normalize_answer(query, candidate)

            proper_noun = re.search(r"\b([A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+){1,6})\b", sentence)
            if proper_noun:
                candidate = proper_noun.group(1).strip()
                return normalize_answer(query, candidate)

    return NOT_FOUND


if __name__ == "__main__":
    mock_chunks = [
        {"doc_id": "test.pdf", "text": "The MSAII program at CMU is known for producing top-tier AI engineers."}
    ]
    test_query = "What CMU program produces top-tier AI engineers?"

    # Local dry run for normalization/fallback behavior.
    print(extract_from_context(test_query, mock_chunks))
