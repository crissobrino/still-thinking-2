import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Importaciones de tu estructura
from scripts.search_chroma import search
from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt, build_system_prompt, build_summarize_prompt
from llm.guardrails import should_refuse
import langid
from deep_translator import GoogleTranslator
from contextlib import asynccontextmanager

def call_llama_uc3m(original_query, context_en, lang_code):
    # Mapeo de códigos a nombres para el modelo
    iso_to_name = {'es': 'Spanish', 'en': 'English', 'fr': 'French', 'de': 'German'}
    target_lang = iso_to_name.get(lang_code, 'English')

    system_prompt = f"""
    You are an expert academic research assistant.
    
    TASK:
    1. Analyze the researcher's idea: "{original_query}"
    2. Compare it with the provided research papers (Context).
    3. Highlight key differences and similarities.

    CONSTRAINTS:
    - You MUST respond entirely in {target_lang}.
    - The provided context is in English, but your analysis must be in {target_lang}.
    - Keep paper titles in their original language.
    - If no relevant papers are found, state exactly: "I’m sorry, but I do not have any additional specific articles in the current corpus on this topic." (Translated to {target_lang}).
    - DO NOT hallucinate.
    """
    
    # Aquí haces el fetch a https://yiyuan.tsc.uc3m.es/api/generate
    
def assistant_logic(user_query):
    # 1. DETECTAR IDIOMA (Instantáneo)
    # langid.classify devuelve ('es', -123.45)
    lang, _ = langid.classify(user_query) 
    
    # 2. TRADUCIR PARA CHROMADB (Solo si no es inglés)
    query_for_search = user_query
    if lang != 'en':
        query_for_search = GoogleTranslator(source='auto', target='en').translate(user_query)
    
    # 3. RECUPERACIÓN (ChromaDB ahora recibe inglés, mayor precisión)
    # results = collection.query(query_texts=[query_for_search], n_results=5)
    context_en = " ".join([res for res in results['documents'][0]])

    # 4. GENERACIÓN FINAL CON LLAMA
    # Le pasamos el contexto en inglés pero le ordenamos responder en el idioma original
    final_response = call_llama_uc3m(user_query, context_en, lang)
    
    return final_response

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

from deep_translator import GoogleTranslator
from llm.language import detect_language, get_language_name

@app.on_event("startup")
async def startup_event():
    print("🚀 Cargando modelos y base de datos... Por favor, espera.")
    # Forzamos una búsqueda vacía o simplemente inicializamos el cliente
    # Esto descargará el modelo de HuggingFace SI NO ESTÁ ya descargado
    # y lo subirá a la RAM.
    try:
        search("warmup", k=1)
        print("System ready. Models loaded in RAM.")
    except Exception as e:
        print(f" Error during pre-load: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- CÓDIGO DE STARTUP ---
    print(" Iniciando Warm-up del sistema...")
    
    # 1. Cargar el modelo de Embeddings (Chroma)
    # Esto descargará/cargará all-MiniLM-L6-v2 antes de que el usuario entre
    try:
        search("warmup query", k=1)
        print(" Modelo de Embeddings cargado.")
    except Exception as e:
        print(f" Error cargando embeddings: {e}")

    # 2. Opcional: Warm-up del LLM (UC3MClient)
    # Las conexiones HTTPS iniciales suelen ser lentas. 
    # Puedes hacer una llamada mínima para "despertar" la conexión.
    try:
        client.chat("ping") 
        print(" Conexión con LLM establecida.")
    except: pass

    yield
    # --- CÓDIGO DE SHUTDOWN (Si fuera necesario) ---
    print("👋 Cerrando servidor...")

app = FastAPI(lifespan=lifespan)

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    # 1. DETECCIÓN DE IDIOMA Y TRADUCCIÓN PARA BÚSQUEDA
    user_lang_code = detect_language(request.query)
    target_language_name = get_language_name(user_lang_code)
    
    # Traducimos la query al inglés para que ChromaDB encuentre mejores resultados
    query_for_chroma = request.query
    if user_lang_code != 'en':
        query_for_chroma = GoogleTranslator(source='auto', target='en').translate(request.query)

    # 2. RECUPERACIÓN DE CHROMA (Usando la query en inglés)
    results = search(query_for_chroma, request.k)
    
    retrieved_docs = []
    # ... (Tu lógica de procesamiento de 'results' se mantiene igual)
    if isinstance(results, list):
        for item in results:
            score = 1 - item.get('distance', 0)
            retrieved_docs.append({
                "title": item.get('title', 'Sin título'),
                "abstract": item.get('abstract', item.get('document', '')),
                "score": score,
                "year": item.get('update_date', 'Desconocido'),
                "authors": item.get('authors', 'Desconocidos')
            })
    else:
        # Lógica para dict crudo
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

    # 3. GUARDRAILS (Anti-alucinación de la rúbrica)
    scores = [doc["score"] for doc in retrieved_docs]
    if should_refuse(retrieved_docs, scores):
        # Mensaje obligatorio según la rúbrica, traducido al idioma del usuario
        refusal_msg = "I’m sorry, but I do not have any additional specific articles in the current corpus on this topic."
        if user_lang_code != 'en':
            refusal_msg = GoogleTranslator(source='en', target=user_lang_code).translate(refusal_msg)
        
        return {
            "answer": refusal_msg,
            "articles": [],
            "language": user_lang_code
        }

    # 4. GENERACIÓN CON LLAMA (Explicándole el idioma de respuesta)
    context = ""
    for i, doc in enumerate(retrieved_docs, start=1):
        context += f"\n[Article {i}]\nTitle: {doc['title']}\nAbstract: {doc['abstract']}\n"

    prompt = build_comparison_prompt(request.query, context, target_language_name)
    system_instruction = build_system_prompt(target_language_name)

    llm_response = client.chat(
        user_prompt=prompt,
        system_prompt=system_instruction,
    )

    # 5. FORMATEO PARA FRONTEND
    articles_for_frontend = []
    for doc in retrieved_docs:
        articles_for_frontend.append({
            "title": doc["title"],
            "authors": doc["authors"],
            "year": doc['year'],
            "abstract": doc["abstract"],  # <--- ¡ESTA LÍNEA ES CRUCIAL!
            "relevanceScore": round(doc["score"] * 100, 1),
            "keyDifference": "..." 
        })

    return {
        "answer": llm_response,
        "articles": articles_for_frontend,
        "language": user_lang_code
    }

class SummarizeRequest(BaseModel):
    abstract: str
    query: str
    language: str = "Spanish"

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    prompt = build_summarize_prompt(request.query, request.abstract, request.language)
    summary = client.chat(user_prompt=prompt, system_prompt="You are a concise academic summarizer.")
    return {"summary": summary}