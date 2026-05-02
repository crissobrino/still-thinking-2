import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path

# --- CONFIGURACIÓN ---
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
        "eval_mode": True, # Activamos para obtener también datos de ENN si hiciera falta
        "use_reranker": use_reranker,
        "use_powerful_model": use_powerful
    }
    try:
        response = requests.post(API_URL, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error llamando a la API: {e}")
        return None

def calculate_metrics(gold_ids, test_ids, gold_scores):
    """
    Compara la lista de IDs del modelo Test contra el Gold Standard.
    """
    if not gold_ids or not test_ids:
        return 0, 0, 0

    # 1. Precision@K (¿Cuántos del test están en el gold?)
    intersection = set(gold_ids) & set(test_ids)
    precision = round(len(intersection) / len(gold_ids), 4)

    # 2. MRR (¿En qué posición está el #1 del Gold en nuestra lista Test?)
    target_id = gold_ids[0]
    mrr = 0
    if target_id in test_ids:
        rank = test_ids.index(target_id) + 1
        mrr = round(1 / rank, 4)

    # 3. NDCG@K (Calidad del ranking)
    # Atribuimos relevancia según la posición en el Gold (5, 4, 3, 2, 1)
    relevance_map = {id: (len(gold_ids) - i) for i, id in enumerate(gold_ids)}
    
    # DCG del Test
    dcg = 0
    for i, tid in enumerate(test_ids):
        rel = relevance_map.get(tid, 0)
        dcg += rel / np.log2(i + 2)
    
    # IDCG (El mejor orden posible, que es el del Gold)
    idcg = 0
    for i in range(len(gold_ids)):
        rel = len(gold_ids) - i
        idcg += rel / np.log2(i + 2)
    
    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0
    
    return precision, mrr, ndcg

def main():
    print(f"🚀 Iniciando evaluación de {len(QUERIES)} consultas...")
    all_results = []

    for q in QUERIES:
        print(f"🔍 Evaluando: {q}")
        
        # 1. Obtener el Gold Standard (Potente + Reranker)
        gold_data = get_api_response(q, use_powerful=True, use_reranker=True)
        # 2. Obtener el Candidato (Rápido + Reranker)
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

    # Crear DataFrame y guardar
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="NDCG@5", ascending=False)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("RESUMEN FINAL")
    print(df[["Query", "NDCG@5", "MRR"]].to_string(index=False))
    print(f"\nMedia NDCG: {df['NDCG@5'].mean():.4f}")
    print(f"Resultados exportados a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()