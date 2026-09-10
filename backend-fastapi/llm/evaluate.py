"""
Evaluation script for the full LLM + ChromaDB pipeline.

Runs a set of test queries against the real backend, scores each response,
and saves a Markdown report to notes/evaluation_report.md.

Scores computed per response:
  - retrieved_k         : number of papers actually returned
  - avg_similarity      : mean similarity score of retrieved papers
  - max_similarity      : best similarity score
  - refused             : True if guardrails blocked the response
  - title_hallucination : True if model output contains a title NOT in retrieved docs
  - has_think_block     : True if model exposed <think> reasoning (qwen3 artefact)
  - think_hallucination : True if <think> block references titles absent from context
  - response_len        : character length of final response (excl. <think>)
  - query_translated    : True if query was translated to English before searching
"""

from __future__ import annotations

import re
import sys
import requests
from pathlib import Path
from datetime import datetime

from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt
from llm.guardrails import should_refuse


BACKEND_URL = "http://localhost:8000/search"
DEFAULT_REPORT_PATH = "notes/evaluation_report.md"

# Reuse the LLM client to translate non-English queries before backend search.
# The ChromaDB embeddings were built on English text, so non-English queries
# return artificially low similarity scores without this step.
TRANSLATE_PROMPT = (
    "Translate the following research query to English. "
    "Return only the translated text, nothing else.\n\nQuery: {query}"
)

TEST_QUERIES = [
    # (label, query, k, expected_behaviour)
    ("Clear match – BERT fine-tuning",
     "fine-tuning BERT for text classification tasks", 5, "should answer"),

    ("Clear match – LLM summarization",
     "large language models for abstractive text summarization", 5, "should answer"),

    ("Clear match – Spanish query (translation fix)",
     "aprendizaje automático para clasificación de texto", 5, "should translate and answer in Spanish"),

    ("Clear match – Spanish NLP (translation fix)",
     "redes neuronales para procesamiento del lenguaje natural", 5, "should translate and answer in Spanish"),

    ("Clear match – knowledge graphs",
     "knowledge graph embeddings for link prediction", 5, "should answer"),

    ("Clear match – sentiment analysis",
     "sentiment analysis using deep learning", 5, "should answer"),

    ("Borderline – broad topic",
     "deep learning", 5, "may answer with generic papers"),

    ("Weak match – unrelated domain (threshold check)",
     "quantum computing error correction", 5, "should now refuse with threshold 0.35"),

    ("Weak match – very off-topic",
     "history of the Roman Empire", 5, "should refuse"),

    ("Edge case – single word",
     "attention", 3, "should answer — attention is a core NLP concept"),
]


def translate_to_english(query: str, client: UC3MClient) -> str:
    """Return English translation of query if it is not already English."""
    lang = detect_language(query)
    if lang.lower() == "english":
        return query
    translated = client.chat(
        user_prompt=TRANSLATE_PROMPT.format(query=query),
        system_prompt="You are a translation assistant.",
        temperature=0.0,
    )
    # strip any <think> block the model may add
    _, clean = strip_think(translated)
    return clean.strip()


def fetch_papers(query: str, k: int = 5) -> list[dict]:
    if not query.strip():
        return []
    try:
        r = requests.post(BACKEND_URL, json={"query": query, "k": k}, timeout=30)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title": doc["title"],
                "abstract": doc["abstract"],
                "score": round(doc["score"], 4),
            }
            for doc in articles
        ]
    except Exception as e:
        print(f"  [backend error] {e}")
        return []


def strip_think(text: str) -> tuple[str, str]:
    """Return (think_block, clean_response) separated from model output."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return think, clean


def _is_known_title(candidate: str, known_titles: set[str]) -> bool:
    """Return True if candidate is a substring of any known title or vice versa."""
    c = candidate.lower().strip()
    for title in known_titles:
        if c in title or title in c:
            return True
    return False


def check_title_hallucination(text: str, retrieved_docs: list[dict]) -> bool:
    """Return True if text contains a quoted title clearly absent from retrieved_docs.

    Uses substring matching to tolerate abbreviated or truncated titles that
    the model produces from the context (e.g. 'Differentially Private Feder.
    Learning' matching the full title).
    """
    quoted = re.findall(r'"([^"]{10,})"', text)
    known = {d["title"].lower() for d in retrieved_docs}
    for candidate in quoted:
        if not _is_known_title(candidate, known):
            return True
    return False


def format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\nSimilarity: {doc['score']}"
        )
    return "\n\n".join(parts)


def run_evaluation() -> list[dict]:
    client = UC3MClient()
    results = []

    for label, query, k, expected in TEST_QUERIES:
        print(f"Running: {label!r} ...")

        # Translate non-English queries so ChromaDB similarity scores are meaningful
        search_query = translate_to_english(query, client) if query.strip() else query
        translated = search_query != query
        if translated:
            print(f"  [translated] {query!r} -> {search_query!r}")

        retrieved = fetch_papers(search_query, k)
        scores = [d["score"] for d in retrieved]
        refused = should_refuse(retrieved, scores)

        if refused:
            think, clean_response = "", ""
        else:
            language = detect_language(query) if query.strip() else "English"
            context = format_context(retrieved)
            prompt = build_comparison_prompt(query, context, language)
            raw_response = client.chat(
                user_prompt=prompt,
                system_prompt="You are a precise academic comparison assistant.",
                temperature=0.0,
            )
            think, clean_response = strip_think(raw_response)

        title_halluc = check_title_hallucination(clean_response, retrieved) if not refused else False
        think_halluc = check_title_hallucination(think, retrieved) if think else False

        result = {
            "label": label,
            "query": query,
            "query_translated": translated,
            "search_query": search_query,
            "expected": expected,
            "retrieved_k": len(retrieved),
            "avg_similarity": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "max_similarity": round(max(scores), 4) if scores else 0.0,
            "refused": refused,
            "title_hallucination": title_halluc,
            "has_think_block": bool(think),
            "think_hallucination": think_halluc,
            "response_len": len(clean_response),
            "retrieved_titles": [d["title"] for d in retrieved],
            "response": clean_response,
        }
        results.append(result)
        status = "REFUSED" if refused else ("HALLUC?" if title_halluc else "OK")
        print(f"  -> {status} | retrieved={len(retrieved)} | max_sim={result['max_similarity']}")

    return results


def save_report(results: list[dict], report_path: str = DEFAULT_REPORT_PATH) -> None:
    Path("notes").mkdir(exist_ok=True)
    lines = [
        "# Pipeline Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary table",
        "",
        "| # | Test | Retrieved | Max sim | Refused | Title halluc | Think halluc | Translated | Resp len |",
        "|---|------|-----------|---------|---------|--------------|--------------|------------|----------|",
    ]
    for i, r in enumerate(results, 1):
        row = (
            f"| {i} | {r['label']} | {r['retrieved_k']} | {r['max_similarity']} "
            f"| {'YES' if r['refused'] else 'no'} "
            f"| {'YES ⚠' if r['title_hallucination'] else 'no'} "
            f"| {'YES ⚠' if r['think_hallucination'] else 'no'} "
            f"| {'yes' if r['query_translated'] else 'no'} "
            f"| {r['response_len']} |"
        )
        lines.append(row)

    lines += ["", "---", "", "## Detailed results", ""]

    for i, r in enumerate(results, 1):
        lines += [
            f"### {i}. {r['label']}",
            f"**Query:** `{r['query']}`  ",
            *([ f"**Translated for search:** `{r['search_query']}`  " ] if r["query_translated"] else []),
            f"**Expected:** {r['expected']}  ",
            f"**Retrieved papers ({r['retrieved_k']}):**",
        ]
        for t in r["retrieved_titles"]:
            lines.append(f"- {t}")
        lines += [
            f"",
            f"**Scores:** avg={r['avg_similarity']} | max={r['max_similarity']}  ",
            f"**Refused:** {r['refused']}  ",
            f"**Title hallucination detected:** {r['title_hallucination']}  ",
            f"**Think-block hallucination:** {r['think_hallucination']}  ",
            "",
            "**Model response:**",
            "```text",
            r["response"] if r["response"] else "(no response — refused)",
            "```",
            "",
        ]

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPORT_PATH
    results = run_evaluation()
    save_report(results, report_path)
