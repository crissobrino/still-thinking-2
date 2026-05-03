"""
Compares MiniLM (chroma_db) vs Specter2 (chroma_db_specter) ANN retrieval.
Gold standard: MiniLM ENN (exact brute-force search).
Metrics: Recall@K, NDCG@K, Latency.

Run from repo root:
    python evaluation/compare_databases.py
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator

HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

from scripts.search_chroma import search_ann as _minilm_ann, search_enn as _minilm_enn
from specter.search_chroma import search_ann as _specter_ann

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

_SPANISH = {
    "aprendizaje automático para clasificación de texto",
    "redes neuronales para procesamiento del lenguaje natural",
}


def _to_english(q: str) -> str:
    if q in _SPANISH:
        return GoogleTranslator(source='auto', target='en').translate(q)
    return q


def _ids(raw: dict) -> list:
    return raw.get('ids', [[]])[0]


def recall(gold: list, test: list) -> float:
    if not gold:
        return 0.0
    return round(len(set(gold) & set(test)) / len(gold), 4)


def ndcg(gold_ids: list, test_ids: list) -> float:
    if not gold_ids or not test_ids:
        return 0.0
    relevance = {gid: len(gold_ids) - i for i, gid in enumerate(gold_ids)}
    dcg  = sum(relevance.get(tid, 0) / np.log2(i + 2) for i, tid in enumerate(test_ids))
    idcg = sum((len(gold_ids) - i) / np.log2(i + 2) for i in range(len(gold_ids)))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def run():
    rows = []
    print(f"Comparing MiniLM vs Specter2 — K={K}\n")

    for q in QUERIES:
        print(f"Query: {q[:70]}")
        q_en = _to_english(q)

        # Gold standard: MiniLM exact nearest neighbours
        t0 = time.perf_counter()
        gold_ids = _ids(_minilm_enn(q_en, K))
        enn_ms = round((time.perf_counter() - t0) * 1000, 2)

        # MiniLM ANN
        t0 = time.perf_counter()
        minilm_ids = _ids(_minilm_ann(q_en, K))
        minilm_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Specter2 ANN
        t0 = time.perf_counter()
        specter_ids = _ids(_specter_ann(q_en, K))
        specter_ms = round((time.perf_counter() - t0) * 1000, 2)

        for name, ids, latency in [
            ("MiniLM ANN",   minilm_ids,  minilm_ms),
            ("Specter2 ANN", specter_ids, specter_ms),
            ("ENN (gold)",   gold_ids,    enn_ms),
        ]:
            r = 1.0 if name == "ENN (gold)" else recall(gold_ids, ids)
            n = 1.0 if name == "ENN (gold)" else ndcg(gold_ids, ids)
            print(f"  {name:15s}  Recall={r:.4f}  NDCG={n:.4f}  {latency}ms")
            rows.append({
                "Query":      q,
                "Config":     name,
                "Recall":     r,
                "NDCG":       n,
                "Latency_ms": latency,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "db_comparison_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    # Recall + NDCG side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["Recall", "NDCG"]):
        sns.barplot(x="Config", y=metric, data=df, capsize=.1, errorbar=("ci", 95), ax=ax)
        ax.set_title(f"{metric}@{K} by Database (CI 95%)")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(f"{metric}@{K}")
    plt.tight_layout()
    chart1 = os.path.join(HERE, "db_comparison_chart.png")
    plt.savefig(chart1)
    plt.close()

    # Latency vs NDCG scatter
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="Latency_ms", y="NDCG", hue="Config", s=150, data=df)
    plt.title(f"Trade-off: Latency vs NDCG@{K}")
    plt.xlabel("Latency (ms)")
    chart2 = os.path.join(HERE, "db_comparison_latency.png")
    plt.savefig(chart2)
    plt.close()

    print(f"Charts saved to {chart1} and {chart2}")

    print("\n--- AGGREGATE ---")
    print(df.groupby("Config")[["Recall", "NDCG"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
