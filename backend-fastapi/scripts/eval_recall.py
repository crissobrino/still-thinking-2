import sys
sys.path.append("/export/data_ml4ds/Neurocosas/others/nlp/Still_thinking/backend-fastapi")
from scripts.search_chroma import search_ann, search_enn

def calculate_recall(query: str, k: int = 10):
    # Obtener resultados
    res_ann = search_ann(query, k=k)
    res_enn = search_enn(query, k=k)

    ids_ann = set(res_ann["ids"][0])
    ids_enn = set(res_enn["ids"][0])

    # Intersección
    intersection = ids_ann.intersection(ids_enn)
    
    recall = len(intersection) / k
    
    print(f"--- Evaluación para: '{query}' (k={k}) ---")
    print(f"ANN IDs: {ids_ann}")
    print(f"ENN IDs: {ids_enn}")
    print(f"Recall: {recall:.2%}")
    
    return recall

if __name__ == "__main__":
    calculate_recall("genomics and machine learning", k=5)