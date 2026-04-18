from pathlib import Path
from datetime import datetime

from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt
from llm.guardrails import should_refuse


def format_context(retrieved_docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        parts.append(
            f"""[Article {i}]
Title: {doc.get('title', 'N/A')}
Abstract: {doc.get('abstract', 'N/A')}
Similarity: {doc.get('score', 'N/A')}"""
        )
    return "\n\n".join(parts)


def get_default_refusal(language: str) -> str:
    if language.lower() == "spanish":
        return (
            "Lo siento, pero no he encontrado artículos suficientemente relevantes "
            "en el corpus actual para comparar con tu línea de investigación."
        )
    return (
        "I’m sorry, but I could not find sufficiently relevant articles "
        "in the current corpus to compare with your research direction."
    )


def run_single_test(client: UC3MClient, test_case: dict) -> dict:
    query = test_case["query"]
    retrieved_docs = test_case["retrieved_docs"]
    scores = [doc.get("score", 0.0) for doc in retrieved_docs]
    language = detect_language(query)

    if should_refuse(retrieved_docs, scores):
        response = get_default_refusal(language)
    else:
        context = format_context(retrieved_docs)
        prompt = build_comparison_prompt(query, context, language)
        response = client.chat(
            user_prompt=prompt,
            system_prompt="You are a precise academic comparison assistant.",
            temperature=0.0,
        )

    return {
        "name": test_case["name"],
        "query": query,
        "language": language,
        "response": response,
        "expected": test_case.get("expected", ""),
        "notes": test_case.get("notes", ""),
    }


def save_results_to_markdown(results: list[dict], output_path: str) -> None:
    lines = []
    lines.append(f"# Prompt experiment log\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")

    for result in results:
        lines.append(f"## {result['name']}\n")
        lines.append(f"**Query:** {result['query']}\n")
        lines.append(f"**Detected language:** {result['language']}\n")
        if result["expected"]:
            lines.append(f"**Expected behavior:** {result['expected']}\n")
        if result["notes"]:
            lines.append(f"**Notes:** {result['notes']}\n")
        lines.append("**Model response:**\n")
        lines.append("```text")
        lines.append(result["response"])
        lines.append("```\n")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    client = UC3MClient()

    test_cases = [
        {
            "name": "Test 1 - Clear match",
            "query": "I am researching federated neural topic models",
            "retrieved_docs": [
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
            ],
            "expected": "Should compare the query with both papers and avoid introducing unsupported technical details.",
        },
        {
            "name": "Test 2 - Weak match",
            "query": "I am studying reinforcement learning for robotics",
            "retrieved_docs": [
                {
                    "title": "Federated topic modeling",
                    "abstract": "This paper studies federated approaches to topic modeling using probabilistic models.",
                    "score": 0.18,
                },
                {
                    "title": "Federated non-negative matrix factorization for short text topic modeling",
                    "abstract": "This paper explores federated NMF methods for short-text topic discovery.",
                    "score": 0.14,
                },
            ],
            "expected": "Should refuse because the retrieved context is not relevant enough.",
        },
        {
            "name": "Test 3 - Spanish query",
            "query": "Estoy investigando modelos de tópicos federados",
            "retrieved_docs": [
                {
                    "title": "Federated topic modeling",
                    "abstract": "This paper studies federated approaches to topic modeling using probabilistic models.",
                    "score": 0.79,
                }
            ],
            "expected": "Should answer in Spanish and keep the article title unchanged.",
        },
        {
            "name": "Test 4 - Empty context",
            "query": "I am researching clinical NLP summarization",
            "retrieved_docs": [],
            "expected": "Should refuse cleanly.",
        },
    ]

    results = [run_single_test(client, test_case) for test_case in test_cases]
    save_results_to_markdown(results, "notes/week1_prompt_tests.md")
    print("Saved results to notes/week1_prompt_tests.md")


if __name__ == "__main__":
    main()