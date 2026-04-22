from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt
from llm.guardrails import should_refuse

import requests

def fetch_papers(query: str, k: int = 5) -> list[dict]:
    response = requests.post("http://localhost:8000/search", json={"query": query, "k": k})
    results = response.json()["results"]
    return [
        {
            "title": doc["title"],
            "abstract": doc["abstract_snippet"],
            "score": 1 - doc["distance"],  # convert distance → similarity
        }
        for doc in results
    ]


# 1. User writes query
query = input("Enter your research query: ").strip()

# 2. Temporary fake retrieved documents
retrieved_docs = fetch_papers(query)
# rest of your pipeline (guardrails, language detection, prompt, LLM) stays the same

scores = [doc["score"] for doc in retrieved_docs]

# 3. Refusal check
if should_refuse(retrieved_docs, scores):
    print("I’m sorry, but I could not find sufficiently relevant articles in the current corpus.")
else:
    # 4. Detect language
    language = detect_language(query)

    # 5. Build context string
    context = ""
    for i, doc in enumerate(retrieved_docs, start=1):
        context += f"""
[Article {i}]
Title: {doc['title']}
Abstract: {doc['abstract']}
Similarity: {doc['score']}
"""

    # 6. Build prompt
    prompt = build_comparison_prompt(query, context, language)

    # 7. Call model
    client = UC3MClient()
    response = client.chat(
        user_prompt=prompt,
        system_prompt="You are a precise academic comparison assistant.",
    )

    # 8. Show response
    print("\n--- MODEL RESPONSE ---\n")
    print(response)