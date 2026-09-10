import time
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

from scripts.search_chroma import search, search_enn
from llm.client import UC3MClient, UC3MClient_large
from llm.language import detect_language, get_language_name
from llm.prompts import build_comparison_prompt, build_system_prompt, build_summarize_prompt
from llm.guardrails import should_refuse
from sentence_transformers import CrossEncoder

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
    # Runs on server startup to pre-load models and avoid cold-start latency on the first request.
    print("Warming up models...")
    dummy_text = "This is a warm-up query."

    # 1. Warm up ChromaDB and the embedding model (all-MiniLM)
    try:
        search(dummy_text, 1)
    except Exception as e:
        print("Warning during ANN warmup:", e)

    # 2. Warm up the re-ranker (CrossEncoder) on the GPU
    reranker.predict([[dummy_text, dummy_text]])

    print("All systems ready. The first search will be immediate.")
    yield
    # Code after yield runs on server shutdown

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
    """Convert raw Chroma output into a clean list of result dictionaries, with safe fallbacks for missing fields."""
    processed = []
    if not results: return processed

    # Case 1: Raw dictionary format (output of search_enn or search_ann)
    if isinstance(results, dict) and 'ids' in results:
        ids = results.get('ids', [[]])[0]

        # Safely extract each field, guarding against absent or empty lists
        documents = results.get('documents', [[]])[0] if results.get('documents') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []

        for i in range(len(ids)):
            # Guard against IndexError if Chroma returns lists of mismatched length
            dist = distances[i] if i < len(distances) else 1.0
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            doc = documents[i] if i < len(documents) and documents[i] is not None else "Abstract not available."

            processed.append({
                "id": ids[i],
                "title": meta.get("title", f"Article {ids[i]}"),
                "abstract": doc,
                "score": 1 - dist,
                "authors": meta.get("authors", "Unknown"),
                "year": meta.get("update_date", "Unknown")
            })

    # Case 2: Pre-processed list format (output of the standard search() function)
    elif isinstance(results, list):
        for item in results:
            processed.append({
                "id": item.get('id'),
                "title": item.get('title', 'Untitled'),
                "abstract": item.get('abstract', item.get('document', '')),
                "score": 1 - item.get('distance', 0),
                "year": item.get('update_date', 'Unknown'),
                "authors": item.get('authors', 'Unknown')
            })
            
    return processed

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    t_start = time.perf_counter()
    metrics = {}
    
    # 1. Translation and language
    user_lang_code = detect_language(request.query)
    target_language_name = get_language_name(user_lang_code)
    
    query_en = request.query
    if user_lang_code != 'en':
        query_en = GoogleTranslator(source='auto', target='en').translate(request.query)

    # Two-stage retrieval: fetch 20 candidates when re-ranking is enabled, otherwise use the requested k.
    initial_k = 20 if request.use_reranker else request.k

    # 2. Initial ANN search
    t0 = time.perf_counter()
    raw_ann = search(query_en, initial_k)
    
    ann_time = (time.perf_counter() - t0) * 1000
    retrieved_ann = process_results(raw_ann)
    extended_articles = retrieved_ann.copy()
    
    metrics = {"ann_time": round(ann_time, 2)}

    # 2.5. Re-ranking and trimming (optional)
    if request.use_reranker and len(retrieved_ann) > 0:
        t_rerank = time.perf_counter()

        # Build query–abstract pairs for the cross-encoder
        pairs = [[query_en, doc["abstract"]] for doc in retrieved_ann]
        scores = reranker.predict(pairs)

        # Update each document's score with the cross-encoder prediction
        for idx, doc in enumerate(retrieved_ann):
            doc["score"] = float(scores[idx])

        # Sort by cross-encoder score, highest first
        retrieved_ann = sorted(retrieved_ann, key=lambda x: x["score"], reverse=True)

        extended_articles = retrieved_ann[:30]

        # Trim to the top-k results
        retrieved_ann = retrieved_ann[:request.k]
        
        metrics["rerank_time"] = round((time.perf_counter() - t_rerank) * 1000, 2)

    t_guardrail_start = time.perf_counter()
    # 3. Guardrails (evaluated on the final top-k document list)
    scores_final = [doc["score"] for doc in retrieved_ann]

    if should_refuse(retrieved_ann, scores_final):
        refusal = "I'm sorry, but I do not have any additional specific articles..."
        if user_lang_code != 'en':
            refusal = GoogleTranslator(source='en', target=user_lang_code).translate(refusal)

        # Record guardrail latency even when the request is refused
        metrics["guardrail_time"] = round((time.perf_counter() - t_guardrail_start) * 1000, 2)
        metrics["total_time"] = round((time.perf_counter() - t_start) * 1000, 2)

        # Zero out token counts to maintain a consistent response structure
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

    # 4. ENN evaluation mode (optional)
    enn_docs = []
    if request.eval_mode:
        t1 = time.perf_counter()
        raw_enn = search_enn(query_en, request.k)  # ENN searches k directly without two-stage retrieval
        enn_time = (time.perf_counter() - t1) * 1000

        enn_docs = process_results(raw_enn)
        
        ids_ann = {d['id'] for d in retrieved_ann if d.get('id')}
        ids_enn = {d['id'] for d in enn_docs if d.get('id')}
        recall = len(ids_ann & ids_enn) / request.k if request.k > 0 else 0
        metrics.update({"enn_time": round(enn_time, 2), "recall": recall})

    # LLM generation
    context = ""
    for i, doc in enumerate(retrieved_ann, start=1):
        context += f"\n[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n"

    metrics["context_tokens"] = len(encoder.encode(context))

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
                llm_res = "Connection error with the advanced model."
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

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    prompt = build_summarize_prompt(request.query, request.abstract, request.language)
    summary = client.chat(user_prompt=prompt, system_prompt="You are a concise academic summarizer.")
    return {"summary": summary}

