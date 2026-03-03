import ollama

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    
    context_text = "\n\n---\n\n".join(
        [f"Source: {chunk['doc_id']}\nText: {chunk['text']}" for chunk in retrieved_chunks]
    )

    system_prompt = f"""You are a precise, highly accurate QA assistant.

PRIORITY RULES:
1) First, try to answer using the provided context (RAG). If the context contains enough info, answer ONLY from it.
2) If the answer is NOT in the context, you MAY use your own general knowledge to answer.
3) If you still don't have a good idea / you are not confident, output exactly: Not Found

FORMAT RULES:
- Keep answers incredibly concise.
- If the answer is a name, date, or single entity, output ONLY that entity.
- Do NOT explain your reasoning.
- Do NOT include citations.

--- Example 1 (Context present) ---
Context:
Source: history.txt
Text: William Pitt the Elder was a British statesman. The city of Pittsburgh was named in his honor.

Question: Who is Pittsburgh named after?
Answer: William Pitt

--- Example 2 (Use model knowledge) ---
Context:
Source: sports.txt
Text: The Pittsburgh Steelers are a professional American football team based in Pittsburgh.

Question: What is the name of the Steelers' head coach?
Answer: Mike Tomlin

--- Example 3 (Truly unknown) ---
Context:
Source: random.txt
Text: This text is irrelevant.

Question: What is the mascot of the secret 2099 Pittsburgh Moon Club?
Answer: Not Found

--- REAL TASK ---

Context:
{context_text}

Question: {query}
Answer:"""

    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': system_prompt
        }
    ])

    answer = response['message']['content'].strip()

    # Normalize exact output for grading
    if answer.lower() in {"not found", "notfound", "not_found", "not-found"}:
        return "Not Found"

    # If the model outputs extra lines, keep it tight
    answer = answer.splitlines()[0].strip()

    # If it returned nothing for some reason
    if not answer:
        return "Not Found"

    return answer

if __name__ == "__main__":
    mock_chunks = [
        {"doc_id": "test.pdf", "text": "The MSAII program at CMU is known for producing top-tier AI engineers."}
    ]
    test_query = "What CMU program produces top-tier AI engineers?"
    
    answer = generate_answer(test_query, mock_chunks)
    print(f"\nLlama 3.1 Answer: {answer}")