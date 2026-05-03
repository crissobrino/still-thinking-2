import requests
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

API_URL = "http://localhost:8000/search"
CHECKPOINT_PATH = "evaluation_recall2/gold_checkpoint.json"
MAX_K = 15
K_VALUES = range(1, MAX_K + 1)

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

def run_optimized_evaluation():
    if not os.path.exists("evaluation_recall2"): os.makedirs("evaluation_recall2")
    
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            checkpoint = json.load(f)
            gold_standards = checkpoint.get("gold_standards", {})
            enn_times = checkpoint.get("enn_times", {})
        print(f"🔄 Checkpoint detectado. Saltando {len(gold_standards)} queries.")
    else:
        gold_standards, enn_times = {}, {}

    # --- PASO 1: ENN (Gold Standard) ---
    print(f"🥇 Paso 1: Generando Gold Standard (ENN @ k={MAX_K})...")
    for q in QUERIES:
        if q in gold_standards: continue
        try:
            # Usamos skip_llm=True para que el Gold Standard sea rápido también
            resp = requests.post(API_URL, json={"query": q, "k": MAX_K, "eval_mode": True, "skip_llm": True}).json()
            gold_ids = [a['id'] for a in resp.get('enn_articles', [])]
            if not gold_ids: continue
            gold_standards[q] = gold_ids
            enn_times[q] = resp.get('metrics', {}).get('enn_time', 0)
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"gold_standards": gold_standards, "enn_times": enn_times}, f)
            print(f"✅ ENN listo: {q[:30]}...")
        except Exception as e:
            print(f"❌ Error en Gold Standard {q}: {e}")

    # --- PASO 2: ANN (Optimizado con Batching y skip_llm) ---
    print(f"\n🚀 Paso 2: Evaluando ANN (Batching + skip_llm)...")
    all_results = []
    total_queries = len(gold_standards)
    start_time = time.time()

    for idx, (q, gold_full) in enumerate(gold_standards.items(), 1):
        try:
            # PEDIMOS MAX_K DE UNA SOLA VEZ
            resp_ann = requests.post(API_URL, json={
                "query": q, 
                "k": MAX_K, 
                "use_reranker": False, 
                "skip_llm": True
            }).json()
            
            # Lista completa de IDs devueltos por ANN
            ids_ann_total = [a['id'] for a in resp_ann.get('articles', [])]
            t_ann_total = resp_ann.get('metrics', {}).get('ann_time', 0)

            # Ahora simulamos cada K localmente
            for k in K_VALUES:
                gold_k = gold_full[:k]
                ids_ann_k = ids_ann_total[:k] # Cogemos los primeros k de la respuesta única
                
                intersection = set(gold_k) & set(ids_ann_k)
                recall = len(intersection) / k if k > 0 else 0

                all_results.append({
                    "Query": q, "K": k, "Recall": recall,
                    "ANN_Latency": t_ann_total, # Latencia de la búsqueda de MAX_K
                    "ENN_Latency": enn_times.get(q, 0)
                })

            # Cálculo de ETA
            elapsed = time.time() - start_time
            avg_per_q = elapsed / idx
            eta = avg_per_q * (total_queries - idx)
            print(f"Progress: [{idx}/{total_queries}] | Last Q: {q[:15]} | ETA: {int(eta//60)}m {int(eta%60)}s", end="\r")

        except Exception as e:
            print(f"\n❌ Error evaluando {q}: {e}")

    # --- FINALIZACIÓN Y GRÁFICAS ---
    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_recall2/fidelity_results_final.csv", index=False)
    
    df_mean = df.groupby('K').mean(numeric_only=True).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(data=df, x="K", y="Recall", marker="o", ax=ax1)
    ax1.set_title("Fidelidad Media (Recall ANN@k vs ENN@k)")
    
    sns.lineplot(data=df_mean, x="K", y="ANN_Latency", label="ANN (batch)", ax=ax2)
    sns.lineplot(data=df_mean, x="K", y="ENN_Latency", label="ENN", ax=ax2)
    ax2.set_title("Latencia Media (ms)")
    
    plt.savefig("evaluation_recall2/fidelity_benchmark.png")
    print(f"\n\n🎉 ¡Proceso finalizado! Resultados en 'evaluation_recall2/'")

if __name__ == "__main__":
    run_optimized_evaluation()