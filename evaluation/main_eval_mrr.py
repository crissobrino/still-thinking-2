import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation_utils import calculate_ndcg

API_URL = "http://localhost:8000/search"

# --- NUEVA FUNCIÓN MRR ---
def calculate_mrr(gold_ids, test_ids):
    """Calcula el Reciprocal Rank para una query."""
    if not gold_ids or not test_ids:
        return 0.0
    gold_set = set(gold_ids)
    for index, doc_id in enumerate(test_ids):
        if doc_id in gold_set:
            return 1.0 / (index + 1)
    return 0.0

def run_main_evaluation():
    with open("evaluation_mrrgold_standard.json", "r") as f:
        gold_standard = json.load(f)
    
    results = []
    print("🚀 Iniciando Evaluación Principal (NDCG + MRR + Latencia)...")

    for q, gold_ids in gold_standard.items():
        # Configuraciones a comparar
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
            
            # Nota: Si es eval_mode para el Gold Standard real de la API, asegúrate 
            # de extraer los IDs de 'enn_articles' si tu API lo separa así. 
            # Aquí mantengo tu lógica original leyendo de 'articles'.
            test_ids = [a['id'] for a in resp.get('articles', [])]
            
            # --- CÁLCULO DE MÉTRICAS ---
            ndcg = calculate_ndcg(gold_ids, test_ids)
            mrr = calculate_mrr(gold_ids, test_ids)
            
            results.append({
                "Query": q,
                "Config": name,
                "NDCG": ndcg,
                "MRR": mrr,
                "Latency": resp['metrics']['total_time']
            })

    # Guardamos los resultados
    df = pd.DataFrame(results)
    df.to_csv("evaluation_mrrresults_main.csv", index=False)

    # --- VISUALIZACIÓN ---
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Killer Chart NDCG
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="NDCG", data=df, capsize=.1, errorbar=('ci', 95), palette="viridis")
    plt.title("Calidad de Ranking Global: NDCG (CI 95%)")
    plt.ylim(0, 1.05)
    plt.savefig("evaluation_mrrkiller_chart_ndcg.png")

    # Gráfico 2: Killer Chart MRR (NUEVO)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Config", y="MRR", data=df, capsize=.1, errorbar=('ci', 95), palette="magma")
    plt.title("Precisión del Top 1: Mean Reciprocal Rank (CI 95%)")
    plt.ylim(0, 1.05)
    plt.savefig("evaluation_mrrkiller_chart_mrr.png")

    # Gráfico 3: Latencia vs NDCG
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Latency", y="NDCG", hue="Config", s=150, data=df, palette="deep")
    plt.title("Trade-off: Latencia vs Calidad (NDCG)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("evaluation_mrrlatency_quality.png")
    
    print("✅ Evaluación completada con éxito.")
    print("\n📊 Resumen de Métricas Promedio:")
    print(df.groupby("Config")[["NDCG", "MRR", "Latency"]].mean())
    print("\nGráficos guardados en la carpeta 'evaluation_mrr'.")

if __name__ == "__main__":
    run_main_evaluation()