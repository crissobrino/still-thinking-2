import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuración
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
    "attention"
]

def calculate_recall(gold_ids, test_ids):
    if not gold_ids: return 0
    intersection = set(gold_ids) & set(test_ids)
    return len(intersection) / len(gold_ids)

def run_recall_evaluation():
    results = []
    print("🚀 Iniciando Evaluación de Recall y Latencia...")

    for q in QUERIES:
        print(f"🔍 Procesando: '{q[:30]}...'")
        
        # 1. Obtenemos el GOLD STANDARD para esta query (Usando ENN)
        # En tu FastAPI, eval_mode=True devuelve enn_articles
        gold_resp = requests.post(API_URL, json={"query": q, "k": K, "eval_mode": True}).json()
        gold_ids = [a['id'] for a in gold_resp.get('enn_articles', [])]
        
        if not gold_ids:
            print(f"⚠️  Ojo: No se encontraron resultados ENN para '{q}'")
            continue

        # 2. Definimos las configuraciones a testear
        # Formato: (Nombre, use_powerful_model, use_reranker, eval_mode)
        configs = [
            ("ANN", False, False, False),
            ("ANN + Rerank", False, True, False),
            ("ENN (Gold)", False, False, True) 
        ]

        for name, pow_model, rerank, eval_m in configs:
            payload = {
                "query": q,
                "k": K,
                "use_powerful_model": pow_model,
                "use_reranker": rerank,
                "eval_mode": eval_m
            }
            
            resp = requests.post(API_URL, json=payload).json()
            
            # Si es la config ENN, sacamos los IDs de 'enn_articles', si no de 'articles'
            if name == "ENN (Gold)":
                test_ids = [a['id'] for a in resp.get('enn_articles', [])]
            else:
                test_ids = [a['id'] for a in resp.get('articles', [])]

            # Calcular métricas
            recall_score = calculate_recall(gold_ids, test_ids)
            latency = resp['metrics']['total_time']

            results.append({
                "Query": q,
                "Config": name,
                "Recall": recall_score,
                "Latency (ms)": latency
            })

    # 3. Procesamiento de Datos
    df = pd.DataFrame(results)
    df.to_csv("recall_results.csv", index=False)
    
    # --- VISUALIZACIÓN ---
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Recall por Configuración (Killer Chart)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Config", y="Recall", data=df, palette="viridis", capsize=.1, errorbar=('ci', 95))
    plt.title(f"Recall@{K}: ANN vs ANN+Rerank vs ENN", fontsize=14)
    plt.ylim(0, 1.1)
    # Añadir etiquetas de valor sobre las barras
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    plt.savefig("recall_killer_chart.png")

    # Gráfico 2: Trade-off Latencia vs Recall
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Latency (ms)", y="Recall", hue="Config", style="Config", s=200, data=df)
    plt.title("Trade-off: Latencia vs Recall", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("latency_recall_tradeoff.png")

    print("\n✅ Evaluación completada.")
    print(f"📊 Recall Promedio por Configuración:\n{df.groupby('Config')['Recall'].mean()}")
    print("\nGráficos guardados como 'recall_killer_chart.png' y 'latency_recall_tradeoff.png'")

if __name__ == "__main__":
    run_recall_evaluation()