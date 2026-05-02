import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_utils import calculate_ndcg

API_URL = "http://localhost:8000/search"

def run_main_evaluation():
    with open("evaluation/gold_standard.json", "r") as f:
        gold_standard = json.load(f)
    
    results = []
    print("🚀 Iniciando Evaluación Principal...")

    for q, gold_ids in gold_standard.items():
        # Configuraciones a comparar
        configs = [
            ("ANN", False, False), 
            ("ANN+Rerank", False, True), 
            ("Gold (ENN+Rerank)", True, True)
        ]
        
        for name, pow, rerank in configs:
            resp = requests.post(API_URL, json={"query": q, "use_powerful_model": pow, "use_reranker": rerank, "eval_mode": (name=="Gold (ENN+Rerank)")}).json()
            
            test_ids = [a['id'] for a in resp.get('articles', [])]
            ndcg = calculate_ndcg(gold_ids, test_ids)
            
            results.append({
                "Query": q,
                "Config": name,
                "NDCG": ndcg,
                "Latency": resp['metrics']['total_time']
            })

    df = pd.DataFrame(results)
    df.to_csv("evaluation/results_main.csv", index=False)

    # Gráfico 1: Killer Chart con Error Bars (Seaborn lo hace solo)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="NDCG", data=df, capsize=.1, errorbar=('ci', 95))
    plt.title("Killer Chart: Calidad por Configuración (CI 95%)")
    plt.savefig("evaluation/killer_chart.png")

    # Gráfico 2: Latencia vs Calidad
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Latency", y="NDCG", hue="Config", s=150, data=df)
    plt.title("Trade-off: Latencia vs Calidad")
    plt.savefig("evaluation/latency_quality.png")
    print("✅ Gráficos principales guardados.")

if __name__ == "__main__":
    run_main_evaluation()