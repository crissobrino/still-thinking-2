import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_utils import calculate_ndcg

API_URL = "http://localhost:8000/search"
OUTPUT_DIR = "evaluation_mrr"

# --- MRR FUNCTION ---
def calculate_mrr(gold_ids, test_ids):
    """Computes the Reciprocal Rank for a single query."""
    if not gold_ids or not test_ids:
        return 0.0
    gold_set = set(gold_ids)
    for index, doc_id in enumerate(test_ids):
        if doc_id in gold_set:
            return 1.0 / (index + 1)
    return 0.0

def run_main_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "gold_standard.json"), "r") as f:
        gold_standard = json.load(f)
    
    results = []
    print("🚀 Starting main evaluation (NDCG + MRR + latency)...")

    for q, gold_ids in gold_standard.items():
        # Configurations to compare
        configs = [
            ("ANN", False, False), 
            ("ANN+Rerank", False, True), 
            ("Gold (ENN+Rerank)", True, True)
        ]
        
        for name, pow, rerank in configs:
            resp = requests.post(API_URL, json={
                "query": q, 
                "use_powerful_model": pow, 
                "use_reranker": rerank, 
                "eval_mode": (name=="Gold (ENN+Rerank)")
            }).json()
            
            # Note: if eval_mode is used for the API's real gold standard, make sure
            # to pull IDs from 'enn_articles' if your API separates them that way.
            # This keeps the original logic of reading from 'articles'.
            test_ids = [a['id'] for a in resp.get('articles', [])]

            # --- METRIC CALCULATION ---
            ndcg = calculate_ndcg(gold_ids, test_ids)
            mrr = calculate_mrr(gold_ids, test_ids)
            
            results.append({
                "Query": q,
                "Config": name,
                "NDCG": ndcg,
                "MRR": mrr,
                "Latency": resp['metrics']['total_time']
            })

    # Save the results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "results_main.csv"), index=False)

    # --- VISUALIZATION ---
    sns.set_theme(style="whitegrid")

    # Chart 1: NDCG bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="NDCG", data=df, capsize=.1, errorbar=('ci', 95), palette="viridis")
    plt.title("Overall Ranking Quality: NDCG (CI 95%)")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(OUTPUT_DIR, "killer_chart_ndcg.png"))

    # Chart 2: MRR bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="MRR", data=df, capsize=.1, errorbar=('ci', 95), palette="magma")
    plt.title("Top-1 Precision: Mean Reciprocal Rank (CI 95%)")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(OUTPUT_DIR, "killer_chart_mrr.png"))

    # Chart 3: latency vs NDCG
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Latency", y="NDCG", hue="Config", s=150, data=df, palette="deep")
    plt.title("Trade-off: Latency vs Quality (NDCG)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_quality.png"))

    print("✅ Evaluation completed successfully.")
    print("\n📊 Mean Metrics Summary:")
    print(df.groupby("Config")[["NDCG", "MRR", "Latency"]].mean())
    print("\nCharts saved in the 'evaluation_mrr' folder.")

if __name__ == "__main__":
    run_main_evaluation()