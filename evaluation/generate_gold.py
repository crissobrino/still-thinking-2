import requests
import json
import os

API_URL = "http://localhost:8000/search"
# Define aquí tus queries de investigación reales
#QUERIES = [
#    "impact of deep learning in medical imaging",
#    "attention mechanisms in transformer models",
#    "quantum error correction code efficiency",
#    "ethical implications of artificial intelligence in hiring",
#    "bert vs roberta for sentiment analysis"
#]

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
    print("🥇 Generando Gold Standard (Configuración Máxima)...")
    
    if not os.path.exists("evaluation"): os.makedirs("evaluation")

    for q in QUERIES:
        try:
            # Usamos eval_mode=True para ENN, use_powerful_model y use_reranker
            payload = {
                "query": q,
                "use_powerful_model": True, 
                "use_reranker": True, 
                "eval_mode": True
            }
            resp = requests.post(API_URL, json=payload).json()
            
            # Guardamos los IDs de los artículos que el sistema "Gold" considera mejores
            gold_ids = [art['id'] for art in resp.get('articles', [])]
            if gold_ids:
                gold_data[q] = gold_ids
                print(f"✅ Gold generado para: {q}")
            else:
                print(f"⚠️ Query '{q}' no devolvió resultados.")
        except Exception as e:
            print(f"❌ Error en query {q}: {e}")

    with open("evaluation/gold_standard.json", "w") as f:
        json.dump(gold_data, f, indent=4)
    print("\n🔥 Archivo 'evaluation/gold_standard.json' creado con éxito.")

if __name__ == "__main__":
    generate_gold()