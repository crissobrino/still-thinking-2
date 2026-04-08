def build_comparison_prompt(query: str, context: str, response_language: str) -> str:
    return f"""
You are an academic research assistant.

Rules:
1. Use only the information in the provided context.
2. Do not invent articles, authors, methods, results, or claims.
3. If the context is empty or insufficient, say so clearly.
4. Respond in {response_language}.
5. Keep article titles exactly as provided.

Task:
Compare the user's proposed research direction with the retrieved articles.
Highlight the main similarities and differences.
Be precise and cautious.

Output format:
- Research direction:
- Relevant articles:
- Key differences:
- Possible novelty:
- Conclusion:

User query:
{query}

Context:
{context}
""".strip()