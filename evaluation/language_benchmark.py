import requests
import json
import csv
import os
from deep_translator import GoogleTranslator

# --- CONFIG ---
API_URL = "http://localhost:8000/search"
FILE_NAME = "full_diagnostic_benchmark3.csv"
LANGUAGES = ['en', 'es', 'fr', 'it', 'de']
QUERIES_BASE = [
    "fine-tuning BERT for text classification tasks",
    "large language models for abstractive text summarization",
    "knowledge graph embeddings for link prediction",
    "sentiment analysis using deep learning",
    "deep learning"
]

def run_evaluation():
    # Include token count columns
    fieldnames = [
        'query_en', 'lang', 'status', 'total_ms', 'ann_ms', 'rerank_ms', 
        'guardrail_ms', 'llm_ms', 'context_tokens', 'answer_tokens',
        'recall_vs_en', 'missing_ids', 'query_translated'
    ]
    
    file_exists = os.path.isfile(FILE_NAME)

    with open(FILE_NAME, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()

        for q_en in QUERIES_BASE:
            group_results = {}
            for lang in LANGUAGES:
                query_tx = q_en if lang == 'en' else GoogleTranslator(source='en', target=lang).translate(q_en)
                try:
                    r = requests.post(API_URL, json={"query": query_tx, "k": 5, "use_reranker": True}, timeout=90)
                    data = r.json()
                    m = data.get("metrics", {})
                    ids = set(str(doc["id"]) for doc in data.get("articles", []))
                    group_results[lang] = {"ids": ids}

                    # Recall analysis
                    recall = 1.0
                    missing = []
                    if lang != 'en' and 'en' in group_results:
                        ids_en = group_results['en']['ids']
                        recall = len(ids & ids_en) / len(ids_en) if ids_en else 0
                        missing = list(ids_en - ids)

                    writer.writerow({
                        'query_en': q_en, 'lang': lang, 'status': 'SUCCESS',
                        'total_ms': m.get('total_time', 0),
                        'ann_ms': m.get('ann_time', 0),
                        'rerank_ms': m.get('rerank_time', 0),
                        'guardrail_ms': m.get('guardrail_time', 0),
                        'llm_ms': m.get('llm_time', 0),
                        'context_tokens': m.get('context_tokens', 0),
                        'answer_tokens': m.get('answer_tokens', 0),
                        'recall_vs_en': round(recall, 2),
                        'missing_ids': json.dumps(missing),
                        'query_translated': query_tx
                    })
                    csvfile.flush()
                    print(f"[{lang.upper()}] Tokens: Ctx={m.get('context_tokens')} Ans={m.get('answer_tokens')} | LLM: {m.get('llm_time')}ms")
                except Exception as e:
                    print(f"❌ Error: {e}")

                print("-" * 50)

if __name__ == "__main__":
    run_evaluation()