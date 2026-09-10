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
        ann_ids = set(search_ann(q, k=k)["ids"][0])  # fast search (HNSW)
        enn_ids = set(search_enn(q, k=k)["ids"][0])  # exact search (brute force)

        recall = len(ann_ids.intersection(enn_ids)) / k
        all_recalls.append(recall)

        print(f"{q[:33]+'...':<35} | {recall:.2%}")

    print("-" * 50)
    print(f"MEAN TOTAL RECALL: {np.mean(all_recalls):.2%}")

if __name__ == "__main__":
    run_benchmark(k=5)