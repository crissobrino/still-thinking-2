import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Importaciones de tu estructura
from scripts.search_chroma import search
from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt
from llm.guardrails import should_refuse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    k: int = 5

client = UC3MClient()

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    # 1. Recuperar de Chroma
    results = search(request.query, request.k)
    
    retrieved_docs = []
    articles_for_frontend = []

    # Verificamos si 'results' es una lista (procesada) o un dict (crudo de Chroma)
    if isinstance(results, list):
        # Si ya es una lista, iteramos directamente
        for item in results:
            # Adaptamos según lo que devuelva tu script 'search_chroma'
            score = 1 - item.get('distance', 0)
            retrieved_docs.append({
                "title": item.get('title', 'Sin título'),
                "abstract": item.get('abstract', item.get('document', '')),
                "score": score,
                "year": item.get('update_date', 'Desconocido'),
                "authors": item.get('authors', 'Desconocidos')
            })
    else:
        # Si es el diccionario crudo de Chroma
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for i in range(len(documents)):
            score = 1 - distances[i]
            retrieved_docs.append({
                "title": metadatas[i].get("title", "Sin título"),
                "abstract": documents[i],
                "score": score,
                "authors": metadatas[i].get("authors", "Desconocidos"),
                "year": metadatas[i].get("update_date", "Desconocido")
            })

    # 2. Guardrails (con print para depurar)
    scores = [doc["score"] for doc in retrieved_docs]
    print(f"DEBUG: Query: {request.query} | Scores: {scores}")

    if should_refuse(retrieved_docs, scores):
        return {
            "answer": "Lo siento, no he encontrado artículos suficientemente relevantes.",
            "articles": [],
            "language": "es"
        }

    # 3. Generación con LLM
    language = detect_language(request.query)
    context = ""
    for i, doc in enumerate(retrieved_docs, start=1):
        context += f"\n[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n"

    prompt = build_comparison_prompt(request.query, context, language)
    
    # Respuesta real del cerebro UC3M
    llm_response = client.chat(
        user_prompt=prompt,
        system_prompt="You are a precise academic comparison assistant.",
    )

    # 4. Formatear artículos para el frontend con datos reales
    for doc in retrieved_docs:
        articles_for_frontend.append({
            "title": doc["title"],
            "authors": doc["authors"], # O mapear desde metadatos
            "year": doc['year'],
            "url": "#",
            "relevanceScore": round(doc["score"] * 100, 1),
            "keyDifference": "Analizado en la respuesta principal." 
        })

    return {
        "answer": llm_response,
        "articles": articles_for_frontend,
        "language": language
    }