import os
from pathlib import Path
from chromadb import PersistentClient
import numpy as np
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))

# Usamos el modelo por defecto de Chroma, pero lo forzamos a la gráfica
try:
    gpu_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cuda" # <--- ¡AQUÍ ESTÁ LA MAGIA DE LA GPU!
    )
    print("Modelo de Embeddings cargado en la GPU (CUDA).")
except Exception as e:
    print(f"Aviso: No se pudo usar CUDA, cayendo a CPU. Error: {e}")
    # Fallback a CPU por si la GPU de la UC3M está saturada por otros alumnos
    gpu_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu" 
    )

client = PersistentClient(path=CHROMA_PATH)

# 👇 3. INYECTAR LA FUNCIÓN EN LA COLECCIÓN 👇
collection = client.get_collection(
    name="papers", 
    embedding_function=gpu_embedding_function
)

def search_ann(query_text: str, k: int = 5):
    """Búsqueda estándar de Chroma (usa HNSW/ANN)"""
    return collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

def search_enn(query_text: str, k: int = 5):
    """Búsqueda Exacta (Fuerza Bruta) obteniendo textos y metadatos"""
    
    all_embeddings = []
    all_ids = []
    all_documents = []
    all_metadatas = []
    
    offset = 0
    batch_size = 5000  # Lote seguro para no saturar SQLite
    
    # 1. Extraemos los datos por lotes
    while True:
        batch = collection.get(
            include=['embeddings', 'documents', 'metadatas'],
            limit=batch_size,
            offset=offset
        )
        
        # Si ya no hay IDs, hemos terminado de leer la base de datos
        if not batch['ids']:
            break
            
        all_ids.extend(batch['ids'])
        all_embeddings.extend(batch['embeddings'])
        
        # Aseguramos que no sean nulos
        docs = batch.get('documents') or []
        metas = batch.get('metadatas') or []
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        
        offset += batch_size

    # 2. Convertimos todo a arrays de Numpy
    np_embeddings = np.array(all_embeddings)
    np_ids = np.array(all_ids)
    np_documents = np.array(all_documents, dtype=object)
    np_metadatas = np.array(all_metadatas, dtype=object)

    embedding_function = collection._embedding_function
    query_embeddings = embedding_function([query_text])
    query_vec = np.array(query_embeddings[0])

    
   # 4. Cálculo de distancia L2 manual (ENN)
    distances = np.linalg.norm(np_embeddings - query_vec, axis=1)
    
    # 5. Obtenemos los índices de los 'k' más cercanos
    idx_sorted = np.argsort(distances)[:k]
    
    # Devolvemos la estructura exacta que espera el main.py
    return {
        "ids": [np_ids[idx_sorted].tolist()],
        "distances": [distances[idx_sorted].tolist()],
        "documents": [np_documents[idx_sorted].tolist()],
        "metadatas": [np_metadatas[idx_sorted].tolist()]
    }

def search(query: str, k: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    output = []

    if not results["ids"] or not results["ids"][0]:
        return output

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i] or {}
        document = results["documents"][0][i] or ""

        output.append({
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", ""),
            "categories": metadata.get("categories", ""),
            "update_date": metadata.get("update_date", ""),
            "document": document,
            "abstract_snippet": document[:500]
        })

    return output


if __name__ == "__main__":
    query = "machine learning for genomics"
    results = search(query)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}]")
        print("ID:", result["id"])
        print("Distance:", result["distance"])
        print("Title:", result["title"])
        print("Authors:", result["authors"])
        print("Categories:", result["categories"])
        print("Update date:", result["update_date"])
        print("Document:", result["abstract_snippet"])