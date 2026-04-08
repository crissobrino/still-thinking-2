from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt
from llm.guardrails import should_refuse

# 1. User writes query
query = input("Enter your research query: ").strip()

# 2. Temporary fake retrieved documents
retrieved_docs = [
    {
        "title": "Federated topic modeling",
        "abstract": "This paper studies federated approaches to topic modeling using probabilistic models.",
        "score": 0.82,
    },
    {
        "title": "Federated non-negative matrix factorization for short text topic modeling",
        "abstract": "This paper explores federated NMF methods for short-text topic discovery.",
        "score": 0.76,
    },
]

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