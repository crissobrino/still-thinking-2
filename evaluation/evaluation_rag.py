import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path

# --- CONFIG ---
API_URL = "http://localhost:8000/search"
OUTPUT_FILE = Path("evaluation/comparativo_modelos.csv")
K_VAL = 5

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
    "attention"
]

def get_api_response(query, use_powerful, use_reranker):
    payload = {
        "query": query,
        "k": K_VAL,
        "eval_mode": True, # also enabled to get ENN data if needed
        "use_reranker": use_reranker,
        "use_powerful_model": use_powerful
    }
    try:
        response = requests.post(API_URL, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error calling the API: {e}")
        return None

def calculate_metrics(gold_ids, test_ids, gold_scores):
    """
    Compares the test model's ID list against the gold standard.
    """
    if not gold_ids or not test_ids:
        return 0, 0, 0

    # 1. Precision@K (how many of the test results are in the gold set?)
    intersection = set(gold_ids) & set(test_ids)
    precision = round(len(intersection) / len(gold_ids), 4)

    # 2. MRR (what rank is the gold #1 result at in the test list?)
    target_id = gold_ids[0]
    mrr = 0
    if target_id in test_ids:
        rank = test_ids.index(target_id) + 1
        mrr = round(1 / rank, 4)

    # 3. NDCG@K (ranking quality)
    # Relevance is assigned by position in the gold list (5, 4, 3, 2, 1)
    relevance_map = {id: (len(gold_ids) - i) for i, id in enumerate(gold_ids)}

    # DCG of the test results
    dcg = 0
    for i, tid in enumerate(test_ids):
        rel = relevance_map.get(tid, 0)
        dcg += rel / np.log2(i + 2)

    # IDCG (best possible ordering, i.e. the gold order)
    idcg = 0
    for i in range(len(gold_ids)):
        rel = len(gold_ids) - i
        idcg += rel / np.log2(i + 2)

    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0

    return precision, mrr, ndcg

def main():
    print(f"🚀 Starting evaluation of {len(QUERIES)} queries...")
    all_results = []

    for q in QUERIES:
        print(f"🔍 Evaluating: {q}")

        # 1. Get the gold standard (powerful model + reranker)
        gold_data = get_api_response(q, use_powerful=True, use_reranker=True)
        # 2. Get the candidate (fast model + reranker)
        test_data = get_api_response(q, use_powerful=False, use_reranker=True)

        if gold_data and test_data:
            gold_ids = [a['id'] for a in gold_data.get('articles', [])]
            test_ids = [a['id'] for a in test_data.get('articles', [])]
            gold_scores = [a['score'] for a in gold_data.get('articles', [])]

            p, mrr, ndcg = calculate_metrics(gold_ids, test_ids, gold_scores)
            
            all_results.append({
                "Query": q,
                "Precision@5": p,
                "MRR": mrr,
                "NDCG@5": ndcg,
                "Time_Gold_ms": gold_data['metrics'].get('total_time'),
                "Time_Test_ms": test_data['metrics'].get('total_time'),
                "Match_Count": len(set(gold_ids) & set(test_ids))
            })

    # Build the DataFrame and save it
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="NDCG@5", ascending=False)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print(df[["Query", "NDCG@5", "MRR"]].to_string(index=False))
    print(f"\nMean NDCG: {df['NDCG@5'].mean():.4f}")
    print(f"Results exported to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()