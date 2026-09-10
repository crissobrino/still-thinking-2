"""
Precision@K evaluation comparing ANN, ANN+Reranking, and ENN configurations.

Requires the backend running at localhost:8000.
Precision@K is measured two ways:
  - score_based: fraction of retrieved docs with similarity >= RELEVANCE_THRESHOLD
  - overlap:     fraction of retrieved docs that also appear in ENN results (gold standard proxy)

Results saved to evaluation/precision_results.csv.
"""

import requests
import pandas as pd
from pathlib import Path

BACKEND_URL = "http://localhost:8000/search"
RELEVANCE_THRESHOLD = 0.35
K = 5
RESULTS_PATH = Path("evaluation/precision_results.csv")

TEST_QUERIES = [
    "fine-tuning BERT for text classification tasks",
    "large language models for abstractive text summarization",
    "aprendizaje automático para clasificación de texto",
    "redes neuronales para procesamiento del lenguaje natural",
    "knowledge graph embeddings for link prediction",
    "sentiment analysis using deep learning",
    "deep learning",
    "quantum computing error correction",
    "history of the Roman Empire",
    "attention",
]


def fetch(query: str, use_reranker: bool, eval_mode: bool) -> dict | None:
    payload = {"query": query, "k": K, "use_reranker": use_reranker, "eval_mode": eval_mode}
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [error] {e}")
        return None


def precision_score_based(articles: list[dict]) -> float:
    if not articles:
        return 0.0
    relevant = sum(1 for a in articles if a.get("score", 0) >= RELEVANCE_THRESHOLD)
    return round(relevant / len(articles), 4)


def precision_overlap(test_ids: list, gold_ids: list) -> float:
    if not gold_ids or not test_ids:
        return 0.0
    return round(len(set(test_ids) & set(gold_ids)) / len(gold_ids), 4)


def run() -> list[dict]:
    rows = []

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")

        # Call 1: ANN + ENN (eval_mode gives us both in one request)
        resp_ann = fetch(query, use_reranker=False, eval_mode=True)

        # Call 2: ANN + Reranking
        resp_rerank = fetch(query, use_reranker=True, eval_mode=False)

        ann_articles    = resp_ann.get("articles", [])       if resp_ann    else []
        enn_articles    = resp_ann.get("enn_articles", [])   if resp_ann    else []
        rerank_articles = resp_rerank.get("articles", [])    if resp_rerank else []

        ann_ids    = [a.get("id") for a in ann_articles]
        enn_ids    = [a.get("id") for a in enn_articles]
        rerank_ids = [a.get("id") for a in rerank_articles]

        row = {
            "query": query,
            # ANN
            "ANN_precision_score":   precision_score_based(ann_articles),
            "ANN_precision_overlap": precision_overlap(ann_ids, enn_ids),
            "ANN_latency_ms":        resp_ann.get("metrics", {}).get("ann_time") if resp_ann else None,
            # ENN (gold standard — score-based only, overlap is 1.0 by definition)
            "ENN_precision_score":   precision_score_based(enn_articles),
            "ENN_latency_ms":        resp_ann.get("metrics", {}).get("enn_time") if resp_ann else None,
            # ANN + Reranking
            "Rerank_precision_score":   precision_score_based(rerank_articles),
            "Rerank_precision_overlap": precision_overlap(rerank_ids, enn_ids),
            "Rerank_latency_ms":        resp_rerank.get("metrics", {}).get("total_time") if resp_rerank else None,
        }
        rows.append(row)

        print(f"  ANN:    P@{K}(score)={row['ANN_precision_score']}, P@{K}(overlap)={row['ANN_precision_overlap']}")
        print(f"  ENN:    P@{K}(score)={row['ENN_precision_score']}")
        print(f"  Rerank: P@{K}(score)={row['Rerank_precision_score']}, P@{K}(overlap)={row['Rerank_precision_overlap']}")

    return rows


def save_csv(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n--- EVALUATION RESULTS ---")
    summary_cols = [
        "query",
        "ANN_precision_score", "ANN_precision_overlap",
        "ENN_precision_score",
        "Rerank_precision_score", "Rerank_precision_overlap",
    ]
    print(df[summary_cols].to_string(index=False))

    print("\n--- MEANS ---")
    for col in ["ANN_precision_score", "ANN_precision_overlap",
                "ENN_precision_score",
                "Rerank_precision_score", "Rerank_precision_overlap"]:
        print(f"  {col}: {df[col].mean():.4f}")

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    results = run()
    save_csv(results)
