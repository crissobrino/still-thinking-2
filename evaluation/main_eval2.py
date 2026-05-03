import sys
import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator

HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

from scripts.search_chroma import search_enn as _search_enn
from evaluate_precision import precision_score_based, precision_overlap

API_URL = "http://localhost:8000/search"
K = 5

QUERIES = [
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

# Queries that need translation before passing to search_enn
_SPANISH = {"aprendizaje automático para clasificación de texto",
            "redes neuronales para procesamiento del lenguaje natural"}


def _to_english(query: str) -> str:
    if query in _SPANISH:
        return GoogleTranslator(source='auto', target='en').translate(query)
    return query


def _enn_articles(query: str) -> tuple[list[dict], float]:
    query_en = _to_english(query)
    t0 = time.perf_counter()
    raw = _search_enn(query_en, K)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    articles = []
    ids      = raw.get("ids", [[]])[0]
    docs     = raw.get("documents", [[]])[0]
    metas    = raw.get("metadatas", [[]])[0]
    dists    = raw.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        articles.append({
            "id":       doc_id,
            "title":    meta.get("title", f"Article {doc_id}"),
            "abstract": docs[i] if i < len(docs) else "",
            "score":    1 - dists[i] if i < len(dists) else 0.0,
            "authors":  meta.get("authors", ""),
            "year":     meta.get("update_date", ""),
        })
    return articles, round(elapsed_ms, 2)


def run_precision_evaluation():
    results = []
    print("Starting Precision@K Evaluation...")

    for q in QUERIES:
        print(f"\nQuery: {q[:70]}")

        resp_base = requests.post(API_URL, json={
            "query": q, "k": K,
            "use_reranker": False,
            "eval_mode": False,
        }, timeout=60).json()

        resp_rerank = requests.post(API_URL, json={
            "query": q, "k": K,
            "use_reranker": True,
            "eval_mode": False,
        }, timeout=60).json()

        ann_articles    = resp_base.get("articles", [])
        rerank_articles = resp_rerank.get("articles", [])
        enn_articles, enn_ms = _enn_articles(q)

        ann_ids    = [a.get("id") for a in ann_articles]
        rerank_ids = [a.get("id") for a in rerank_articles]
        enn_ids    = [a.get("id") for a in enn_articles]

        configs = [
            ("ANN",        ann_articles,    ann_ids,    resp_base.get("metrics",   {}).get("ann_time")),
            ("ANN+Rerank", rerank_articles, rerank_ids, resp_rerank.get("metrics", {}).get("total_time")),
            ("Gold (ENN)", enn_articles,    enn_ids,    enn_ms),
        ]

        for name, articles, ids, latency in configs:
            p_score   = precision_score_based(articles)
            p_overlap = 1.0 if name == "Gold (ENN)" else precision_overlap(ids, enn_ids)

            results.append({
                "Query":             q,
                "Config":            name,
                "Precision_Score":   p_score,
                "Precision_Overlap": p_overlap,
                "Latency":           latency,
            })

            print(f"  {name:14s}  P@{K}(score)={p_score:.4f}  P@{K}(overlap)={p_overlap:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("evaluation/precision_results_main.csv", index=False)
    print("\nResults saved to evaluation/precision_results_main.csv")

    # Killer chart — score-based precision by config
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="Precision_Score", data=df, capsize=.1, errorbar=("ci", 95))
    plt.title(f"Killer Chart: Precision@{K} by Configuration (CI 95%)")
    plt.ylabel(f"Precision@{K} (score-based, threshold=0.35)")
    plt.savefig("evaluation/precision_killer_chart.png")
    plt.close()

    # Latency vs precision
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Latency", y="Precision_Score", hue="Config", s=150, data=df)
    plt.title(f"Trade-off: Latency vs Precision@{K}")
    plt.ylabel(f"Precision@{K} (score-based)")
    plt.savefig("evaluation/precision_latency_quality.png")
    plt.close()

    print("Charts saved.")


if __name__ == "__main__":
    run_precision_evaluation()
