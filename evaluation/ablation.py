import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_utils import calculate_ndcg

API_URL = "http://localhost:8000/search"


def call_rag(query: str, use_powerful_model: bool, use_reranker: bool):
    """Calls the search endpoint and returns (response_json, latency_ms)."""
    resp = requests.post(API_URL, json={
        "query": query,
        "use_powerful_model": use_powerful_model,
        "use_reranker": use_reranker,
    }).json()
    return resp, resp.get("metrics", {}).get("total_time")


configs = [
    {"name": "ANN Only", "pow": False, "rerank": False},
    {"name": "ANN + Reranker", "pow": False, "rerank": True},
    {"name": "ENN + Reranker (Gold)", "pow": True, "rerank": True}
]

ablation_data = []
for q in ["attention mechanism", "bert fine tuning"]:
    # Get the real gold standard first, to compare every config against it
    gold_data, _ = call_rag(q, True, True)
    gold_ids = [a['id'] for a in gold_data.get('articles', [])]

    for cfg in configs:
        data, _ = call_rag(q, cfg['pow'], cfg['rerank'])
        test_ids = [a['id'] for a in data.get('articles', [])]
        ndcg = calculate_ndcg(gold_ids, test_ids)
        ablation_data.append({"Config": cfg['name'], "NDCG": ndcg, "Latency": data['metrics']['total_time']})

df_abl = pd.DataFrame(ablation_data)

# Chart A: retrieval quality by configuration
sns.barplot(x="Config", y="NDCG", data=df_abl, palette="viridis")
plt.title("Ablation: Impact of Architecture on Quality (NDCG@5)")
plt.show()

# Chart B: latency vs quality trade-off
sns.scatterplot(x="Latency", y="NDCG", hue="Config", s=100, data=df_abl)
plt.title("Trade-off: Latency vs Quality")
plt.xlabel("Response Time (ms)")
plt.show()
