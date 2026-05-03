import time
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from contextlib import asynccontextmanager
import torch
import tiktoken
from fastapi.responses import StreamingResponse 
import json

encoder = tiktoken.get_encoding("cl100k_base")

# Ajuste de path e importaciones
sys.path.append("/export/data_ml4ds/Neurocosas/others/nlp/Still_thinking/backend-fastapi")
from scripts.search_chroma import search, search_enn
from llm.client import UC3MClient, UC3MClient_large
from llm.language import detect_language, get_language_name
from llm.prompts import build_comparison_prompt, build_system_prompt, build_summarize_prompt
from llm.guardrails import should_refuse
from sentence_transformers import CrossEncoder
import torch

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', activation_fn=torch.nn.Sigmoid(), device="cuda" if torch.cuda.is_available() else "cpu")
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    eval_mode: bool = False
    use_reranker: bool = False
    use_powerful_model: bool = False
    skip_llm: bool = False
    stream: bool = False

class SummarizeRequest(BaseModel):

    abstract: str
    query: str
    language: str = "Spanish"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esto se ejecuta AL ARRANCAR el servidor
    print("🔥 Calentando motores (Warm-up de Modelos)...")
    dummy_text = "This is a warm-up query."
    
    # 1. Calentamos Chroma y el modelo de Embeddings (all-MiniLM)
    try:
        search(dummy_text, 1) 
    except Exception as e:
        print("Aviso en warmup ANN:", e)
        
    # 2. Calentamos el Re-ranker (CrossEncoder) en la GPU
    reranker.predict([[dummy_text, dummy_text]])
    
    print("✅ ¡Sistemas 100% listos! La primera búsqueda será instantánea.")
    yield
    # (Lo que pongas después del yield se ejecutaría al apagar el servidor)

app = FastAPI(lifespan=lifespan)
client = UC3MClient()

client2 = UC3MClient_large()


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
    t_start = time.perf_counter()
    metrics = {}
    
    # 1. Idioma y Traducción
    user_lang_code = detect_language(request.query)
    target_language_name = get_language_name(user_lang_code)
    
    query_en = request.query
    if user_lang_code != 'en':
        query_en = GoogleTranslator(source='auto', target='en').translate(request.query)

    # 👇 LÓGICA TWO-STAGE: Si usamos re-ranker, pedimos 20 a Chroma. Si no, pedimos los que diga la request (5).
    initial_k = 20 if request.use_reranker else request.k


    # 2. Búsqueda ANN Inicial
    t0 = time.perf_counter()
    raw_ann = search(query_en, initial_k) # Usamos initial_k
    
    ann_time = (time.perf_counter() - t0) * 1000
    retrieved_ann = process_results(raw_ann)
    extended_articles = retrieved_ann.copy()
    
    metrics = {"ann_time": round(ann_time, 2)}

    # 👇 NUEVO: 2.5 Re-ranking y Recorte (Opcional)
    if request.use_reranker and len(retrieved_ann) > 0:
        t_rerank = time.perf_counter()
        
        # Preparamos los pares [Pregunta, Abstract]
        pairs = [[query_en, doc["abstract"]] for doc in retrieved_ann]
        scores = reranker.predict(pairs)
        
        # Actualizamos puntuaciones
        for idx, doc in enumerate(retrieved_ann):
            doc["score"] = float(scores[idx])
            
        # Ordenamos de mayor a menor según la precisión del Cross-Encoder
        retrieved_ann = sorted(retrieved_ann, key=lambda x: x["score"], reverse=True)
        
        extended_articles = retrieved_ann[:30]

        # RECORTE: Nos quedamos estrictamente con los mejores K (5)
        retrieved_ann = retrieved_ann[:request.k]
        
        metrics["rerank_time"] = round((time.perf_counter() - t_rerank) * 1000, 2)

    t_guardrail_start = time.perf_counter()
    # 3. Guardrails (Evalúa sobre la lista final de 5 documentos)
    scores_final = [doc["score"] for doc in retrieved_ann]

    if should_refuse(retrieved_ann, scores_final):
        refusal = "I’m sorry, but I do not have any additional specific articles..."
        if user_lang_code != 'en':
            refusal = GoogleTranslator(source='en', target=user_lang_code).translate(refusal)
        
        # Guardamos latencia del guardrail incluso si bloquea
        metrics["guardrail_time"] = round((time.perf_counter() - t_guardrail_start) * 1000, 2)
        metrics["total_time"] = round((time.perf_counter() - t_start) * 1000, 2)
        
        # 👇 NUEVO: Añadimos las llaves con valor 0 para mantener la estructura
        metrics["context_tokens"] = 0
        metrics["answer_tokens"] = 0
        
        return {
            "answer": refusal, 
            "articles": [], 
            "extended_articles": extended_articles,
            "language": user_lang_code, 
            "metrics": metrics
        }

    metrics["guardrail_time"] = round((time.perf_counter() - t_guardrail_start) * 1000, 2)

    # Filter to only papers above the relevance threshold (same value as the guardrail)
    RELEVANCE_THRESHOLD = 0.35
    retrieved_ann = [doc for doc in retrieved_ann if doc["score"] >= RELEVANCE_THRESHOLD]

    # 4. Modo Evaluación ENN (Opcional)
    enn_docs = []
    if request.eval_mode:
        t1 = time.perf_counter()
        raw_enn = search_enn(query_en, request.k) # ENN siempre busca K directamente
        enn_time = (time.perf_counter() - t1) * 1000

        enn_docs = process_results(raw_enn)
        
        ids_ann = {d['id'] for d in retrieved_ann if d.get('id')}
        ids_enn = {d['id'] for d in enn_docs if d.get('id')}
        recall = len(ids_ann & ids_enn) / request.k if request.k > 0 else 0
        metrics.update({"enn_time": round(enn_time, 2), "recall": recall})

    # 5. Generación LLM
    context = ""
    for i, doc in enumerate(retrieved_ann, start=1):
        context += f"\n[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n"

    # Calculamos tokens del contexto enviado al LLM
    metrics["context_tokens"] = len(encoder.encode(context)) # <--- TOKENS ENTRADA

    if request.skip_llm:
        llm_res = "LLM Generation skipped for evaluation."
        metrics["llm_time"] = 0
        metrics["answer_tokens"] = 0
    else:
        prompt = build_comparison_prompt(request.query, context, target_language_name)
        system_p = build_system_prompt(target_language_name)
        t_llm = time.perf_counter()
        
        if request.use_powerful_model:
            try:
                llm_res = client2.chat(user_prompt=prompt, system_prompt=system_p, temperature=0.3)
            except Exception as e:
                llm_res = "Error de conexión con modelo avanzado."
        else:
            llm_res = client.chat(user_prompt=prompt, system_prompt=system_p)

        metrics["llm_time"] = round((time.perf_counter() - t_llm) * 1000, 2)
        metrics["answer_tokens"] = len(encoder.encode(llm_res))

    total_time = (time.perf_counter() - t_start) * 1000
    metrics["total_time"] = round(total_time, 2)

    return {
        "answer": llm_res,
        "articles": retrieved_ann,
        "extended_articles": extended_articles,
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

