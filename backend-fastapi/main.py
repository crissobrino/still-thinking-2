import time
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from contextlib import asynccontextmanager

# Ajuste de path e importaciones
sys.path.append("/export/data_ml4ds/Neurocosas/others/nlp/Still_thinking/backend-fastapi")
from scripts.search_chroma import search, search_enn
from llm.client import UC3MClient
from llm.language import detect_language, get_language_name
from llm.prompts import build_comparison_prompt, build_system_prompt, build_summarize_prompt
from llm.guardrails import should_refuse

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    eval_mode: bool = False

class SummarizeRequest(BaseModel):

    abstract: str
    query: str
    language: str = "Spanish"

app = FastAPI()
client = UC3MClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def process_results(results):
    """Convierte la salida cruda de Chroma en una lista limpia de diccionarios, a prueba de fallos."""
    processed = []
    if not results: return processed
    
    # CASO 1: Formato de diccionario directo (Output de nuestro nuevo search_enn o search_ann crudo)
    if isinstance(results, dict) and 'ids' in results:
        ids = results.get('ids', [[]])[0]
        
        # Extraemos de forma segura comprobando que las listas existan
        documents = results.get('documents', [[]])[0] if results.get('documents') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        
        for i in range(len(ids)):
            # Evitamos IndexError si por algún fallo de Chroma las listas tienen distinto tamaño
            dist = distances[i] if i < len(distances) else 1.0
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            doc = documents[i] if i < len(documents) and documents[i] is not None else "Abstract no disponible."
            
            processed.append({
                "id": ids[i],
                "title": meta.get("title", f"Artículo {ids[i]}"),
                "abstract": doc,
                "score": 1 - dist,
                "authors": meta.get("authors", "Desconocidos"),
                "year": meta.get("update_date", "Desconocido")
            })
            
    # CASO 2: Formato de lista (Output pre-procesado de tu función 'search()' normal)
    elif isinstance(results, list):
        for item in results:
            processed.append({
                "id": item.get('id'),
                "title": item.get('title', 'Sin título'),
                "abstract": item.get('abstract', item.get('document', '')),
                "score": 1 - item.get('distance', 0),
                "year": item.get('update_date', 'Desconocido'),
                "authors": item.get('authors', 'Desconocidos')
            })
            
    return processed

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    # 1. Idioma y Traducción
    user_lang_code = detect_language(request.query)
    target_language_name = get_language_name(user_lang_code)
    
    query_en = request.query
    if user_lang_code != 'en':
        query_en = GoogleTranslator(source='auto', target='en').translate(request.query)

    # 2. Búsqueda ANN (Siempre se hace)
    t0 = time.perf_counter()
    raw_ann = search(query_en, request.k)
    ann_time = (time.perf_counter() - t0) * 1000
    retrieved_ann = process_results(raw_ann)

    # 3. Guardrails (Usamos resultados ANN para decidir si responder)
    scores = [doc["score"] for doc in retrieved_ann]
    if should_refuse(retrieved_ann, scores):
        refusal = "I’m sorry, but I do not have any additional specific articles in the current corpus on this topic."
        if user_lang_code != 'en':
            refusal = GoogleTranslator(source='en', target=user_lang_code).translate(refusal)
        return {"answer": refusal, "articles": [], "language": user_lang_code, "metrics": {"ann_time": round(ann_time, 2)}}

    # 4. Modo Evaluación (Opcional)
    metrics = {"ann_time": round(ann_time, 2)}
    enn_docs = []
    if request.eval_mode:
        t1 = time.perf_counter()
        raw_enn = search_enn(query_en, request.k)
        enn_time = (time.perf_counter() - t1) * 1000


        ##########################################
        print("\n" + "="*30)
        print("🔍 DEBUG ENN RAW:")
        print(f"Tipo de raw_enn: {type(raw_enn)}")
        print(f"Contenido: {raw_enn}")
        print("="*30 + "\n")







        enn_docs = process_results(raw_enn)
        
        # Calcular Recall
        ids_ann = {d['id'] for d in retrieved_ann if d.get('id')}
        ids_enn = {d['id'] for d in enn_docs if d.get('id')}
        recall = len(ids_ann & ids_enn) / request.k if request.k > 0 else 0
        metrics.update({"enn_time": round(enn_time, 2), "recall": recall})

    # 5. Generación LLM
    context = ""
    for i, doc in enumerate(retrieved_ann, start=1):
        context += f"\n[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n"

    prompt = build_comparison_prompt(request.query, context, target_language_name)
    llm_res = client.chat(user_prompt=prompt, system_prompt=build_system_prompt(target_language_name))

    return {
        "answer": llm_res,
        "articles": retrieved_ann,
        "enn_articles": enn_docs if request.eval_mode else [],
        "metrics": metrics,
        "language": user_lang_code
    }

# Endpoint de resumen se mantiene igual...
@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    prompt = build_summarize_prompt(request.query, request.abstract, request.language)
    summary = client.chat(user_prompt=prompt, system_prompt="You are a concise academic summarizer.")
    return {"summary": summary}