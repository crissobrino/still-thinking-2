import requests
import json
import csv
import os
from deep_translator import GoogleTranslator

API_URL = "http://localhost:8000/search"
FILE_NAME = "deep_analysis_benchmark.csv"
LANGUAGES = ['en', 'es', 'fr', 'it', 'de']
QUERIES_BASE = [
    "fine-tuning BERT for text classification tasks",
    "large language models for abstractive text summarization",
    "knowledge graph embeddings for link prediction",
    "sentiment analysis using deep learning",
    "deep learning"
]

def run_evaluation():
    fieldnames = [
        'query_en', 'lang', 'recall_vs_en', 'missing_ids', 'extra_ids', 
        'llm_ms', 'lang_detected', 'query_translated', 'status'
    ]
    
    file_exists = os.path.isfile(FILE_NAME)

    with open(FILE_NAME, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()

        for q_en in QUERIES_BASE:
            # Diccionario para guardar los resultados del grupo por cada idioma
            group_results = {}
            
            for lang in LANGUAGES:
                query_to_send = q_en if lang == 'en' else GoogleTranslator(source='en', target=lang).translate(q_en)
                
                try:
                    response = requests.post(API_URL, json={
                        "query": query_to_send,
                        "k": 5,
                        "use_reranker": True
                    }, timeout=60)
                    data = response.json()
                    
                    ids = set(str(doc["id"]) for doc in data.get("articles", []))
                    group_results[lang] = {
                        "ids": ids,
                        "llm_ms": data.get("metrics", {}).get("llm_time", 0),
                        "detected": data.get("language"),
                        "query_tx": query_to_send
                    }

                    # --- ANÁLISIS DE FALLO ---
                    recall = 1.0
                    missing = []
                    extra = []
                    
                    if lang != 'en' and 'en' in group_results:
                        ids_en = group_results['en']['ids']
                        # Intersección: lo que está en ambos
                        intersection = ids & ids_en
                        # Recall: (comunes) / (total esperados en inglés)
                        recall = len(intersection) / len(ids_en) if len(ids_en) > 0 else 0
                        # Missing: estaba en inglés pero no en este idioma
                        missing = list(ids_en - ids)
                        # Extra: está en este idioma pero no estaba en el inglés
                        extra = list(ids - ids_en)

                    # --- GUARDADO ---
                    writer.writerow({
                        'query_en': q_en,
                        'lang': lang,
                        'recall_vs_en': round(recall, 2),
                        'missing_ids': json.dumps(missing),
                        'extra_ids': json.dumps(extra),
                        'llm_ms': group_results[lang]['llm_ms'],
                        'lang_detected': group_results[lang]['detected'],
                        'query_translated': query_to_send,
                        'status': 'SUCCESS'
                    })
                    csvfile.flush()

                    print(f"[{lang.upper()}] Q: {q_en[:20]}.. | Recall: {recall:.2f} | Missing: {len(missing)}")

                except Exception as e:
                    writer.writerow({'query_en': q_en, 'lang': lang, 'status': f'ERROR: {str(e)}'})
                    print(f"❌ Error en {lang}: {e}")

if __name__ == "__main__":
    run_evaluation()