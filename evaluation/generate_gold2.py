import requests
import json
import os

API_URL = "http://localhost:8000/search"

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

def generate_gold():
    gold_data = {}
    print("🥇 Generando Gold Standard (Solo Búsqueda Exacta ENN)...")
    
    if not os.path.exists("evaluation"): os.makedirs("evaluation")

    for q in QUERIES:
        try:
            # Para el Gold puro, solo nos importa eval_mode=True
            payload = {
                "query": q,
                "k": 5,
                "use_powerful_model": False, 
                "use_reranker": False, 
                "eval_mode": True
            }
            resp = requests.post(API_URL, json=payload).json()
            
            # IMPORTANTE: Extraemos de 'enn_articles', que es el resultado sin aproximaciones
            gold_ids = [art['id'] for art in resp.get('enn_articles', [])]
            
            if gold_ids:
                gold_data[q] = gold_ids
                print(f"✅ Gold (ENN) generado para: {q}")
            else:
                print(f"⚠️ Query '{q}' no devolvió resultados ENN.")
        except Exception as e:
            print(f"❌ Error en query {q}: {e}")

    with open("evaluation/gold_standard2.json", "w") as f:
        json.dump(gold_data, f, indent=4)
    print("\n🔥 Archivo 'evaluation/gold_standard2.json' (PURE ENN) creado.")

if __name__ == "__main__":
    generate_gold()