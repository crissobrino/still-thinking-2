import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from chromadb import PersistentClient
from specter2_ef import Specter2EmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db_specter"))

_ef = Specter2EmbeddingFunction()
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("papers", embedding_function=_ef)


def search_ann(query_text: str, k: int = 5):
    return collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )


def search_enn(query_text: str, k: int = 5):
    all_embeddings = []
    all_ids = []
    all_documents = []
    all_metadatas = []

    offset = 0
    batch_size = 5000

    while True:
        batch = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        all_embeddings.extend(batch["embeddings"])
        all_documents.extend(batch.get("documents") or [])
        all_metadatas.extend(batch.get("metadatas") or [])
        offset += batch_size

    np_embeddings = np.array(all_embeddings)
    np_ids = np.array(all_ids)
    np_documents = np.array(all_documents, dtype=object)
    np_metadatas = np.array(all_metadatas, dtype=object)

    query_vec = np.array(_ef([query_text])[0])
    distances = np.linalg.norm(np_embeddings - query_vec, axis=1)
    idx_sorted = np.argsort(distances)[:k]

    return {
        "ids": [np_ids[idx_sorted].tolist()],
        "distances": [distances[idx_sorted].tolist()],
        "documents": [np_documents[idx_sorted].tolist()],
        "metadatas": [np_metadatas[idx_sorted].tolist()],
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
            "abstract_snippet": document[:500],
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
