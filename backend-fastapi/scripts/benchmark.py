import sys
sys.path.append("/export/data_ml4ds/Neurocosas/others/nlp/Still_thinking/backend-fastapi")
from scripts.search_chroma import search_ann, search_enn
import numpy as np

test_queries = [
    "deep learning in cancer detection",
    "transformer models for NLP",
    "genomic sequencing tools",
    "transformer algorithms",
    "large language processing in cancer",
    "augmentations for transformers"
]

def run_benchmark(k=10):
    all_recalls = []
    
    print(f"{'Query':<35} | {'Recall':<10}")
    print("-" * 50)
    
    for q in test_queries:
        # IDs de la búsqueda rápida (HNSW)
        ann_ids = set(search_ann(q, k=k)["ids"][0])
        # IDs de la búsqueda real (Fuerza bruta)
        enn_ids = set(search_enn(q, k=k)["ids"][0])
        
        # Cálculo: Intersección / K
        recall = len(ann_ids.intersection(enn_ids)) / k
        all_recalls.append(recall)
        
        print(f"{q[:33]+'...':<35} | {recall:.2%}")
    
    print("-" * 50)
    print(f"RECALL MEDIO TOTAL: {np.mean(all_recalls):.2%}")

if __name__ == "__main__":
    run_benchmark(k=5)