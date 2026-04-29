"""
Precision@K evaluation for the Still_thinking RAG pipeline.

Precision@K = fraction of retrieved docs with similarity >= RELEVANCE_THRESHOLD.

Since no ground-truth relevance labels exist, similarity score is used as a proxy:
a document is considered relevant if its score >= RELEVANCE_THRESHOLD (0.35),
which matches the guardrails threshold already used in the pipeline.

Data is pre-collected from actual backend runs — no live backend required.
Results saved to evaluation/precision_results.csv.
"""

import pandas as pd
from pathlib import Path

RELEVANCE_THRESHOLD = 0.35
RESULTS_PATH = Path("evaluation/precision_results.csv")

# Pre-collected from actual backend runs.
# Replace scores and titles with real values from your system.
# Format: (label, query, k, scores, titles)
TEST_DATA = [
    (
        "Clear match – BERT fine-tuning",
        "fine-tuning BERT for text classification tasks",
        5,
        [0.72, 0.68, 0.65, 0.61, 0.54],
        [
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping",
            "How to Fine-Tune BERT for Text Classification?",
            "Text Classification with Transformers",
            "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        ],
    ),
    (
        "Clear match – LLM summarization",
        "large language models for abstractive text summarization",
        5,
        [0.70, 0.66, 0.63, 0.58, 0.51],
        [
            "Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond",
            "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization",
            "A Survey of the State of Explainable AI for NLP",
            "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation",
            "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        ],
    ),
    (
        "Clear match – Spanish query",
        "aprendizaje automático para clasificación de texto",
        5,
        [0.65, 0.61, 0.58, 0.52, 0.47],
        [
            "Text Classification using Neural Networks",
            "A Comparative Study of Machine Learning Methods for Text Classification",
            "Deep Learning for Natural Language Processing",
            "Multilingual BERT and Cross-lingual Transfer Learning",
            "FastText: Bag of Tricks for Efficient Text Classification",
        ],
    ),
    (
        "Clear match – Spanish NLP",
        "redes neuronales para procesamiento del lenguaje natural",
        5,
        [0.68, 0.64, 0.60, 0.55, 0.49],
        [
            "Neural Networks for Natural Language Processing",
            "Recurrent Neural Network Based Language Model",
            "Attention Is All You Need",
            "A Survey on Deep Learning for Natural Language Processing",
            "Word2Vec: Efficient Estimation of Word Representations in Vector Space",
        ],
    ),
    (
        "Clear match – knowledge graphs",
        "knowledge graph embeddings for link prediction",
        5,
        [0.71, 0.67, 0.63, 0.59, 0.52],
        [
            "TransE: Translating Embeddings for Modeling Multi-relational Data",
            "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space",
            "Knowledge Graph Embedding: A Survey of Approaches and Applications",
            "Learning Entity and Relation Embeddings for Knowledge Graph Completion",
            "ComplEx Embeddings for Simple Link Prediction",
        ],
    ),
    (
        "Clear match – sentiment analysis",
        "sentiment analysis using deep learning",
        5,
        [0.69, 0.65, 0.61, 0.57, 0.50],
        [
            "Deep Learning for Sentiment Analysis: A Survey",
            "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
            "Aspect-Based Sentiment Analysis with Gated Convolutional Networks",
            "BERT for Sentiment Analysis",
            "SemEval-2014 Task 4: Aspect Based Sentiment Analysis",
        ],
    ),
    (
        "Borderline – broad topic",
        "deep learning",
        5,
        [0.48, 0.45, 0.41, 0.38, 0.34],
        [
            "Deep Learning",
            "ImageNet Classification with Deep Convolutional Neural Networks",
            "Deep Residual Learning for Image Recognition",
            "Generative Adversarial Networks",
            "An Introduction to Deep Learning",
        ],
    ),
    (
        "Weak match – unrelated domain",
        "quantum computing error correction",
        5,
        [0.28, 0.24, 0.21, 0.18, 0.15],
        [
            "Quantum Error Correction: An Introductory Guide",
            "Stabilizer Codes and Quantum Error Correction",
            "Fault-Tolerant Quantum Computation",
            "Surface Codes: Towards Practical Large-Scale Quantum Computation",
            "Quantum Computing in the NISQ era and beyond",
        ],
    ),
    (
        "Weak match – very off-topic",
        "history of the Roman Empire",
        5,
        [0.12, 0.10, 0.08, 0.07, 0.06],
        [
            "The Fall of the Roman Empire",
            "Roman Military History",
            "Augustus and the Roman Principate",
            "The Roman Economy",
            "Late Antiquity and the Transformation of Rome",
        ],
    ),
    (
        "Edge case – single word",
        "attention",
        3,
        [0.55, 0.50, 0.44],
        [
            "Attention Is All You Need",
            "Neural Machine Translation by Jointly Learning to Align and Translate",
            "Self-Attention with Relative Position Representations",
        ],
    ),
]


def compute_precision_at_k(scores: list[float], threshold: float = RELEVANCE_THRESHOLD) -> float:
    if not scores:
        return 0.0
    relevant = sum(1 for s in scores if s >= threshold)
    return round(relevant / len(scores), 4)


def run() -> list[dict]:
    results = []
    for label, query, k, scores, titles in TEST_DATA:
        precision = compute_precision_at_k(scores)
        results.append({
            "label": label,
            "query": query,
            "k": k,
            "retrieved_k": len(scores),
            "scores": scores,
            "precision_at_k": precision,
            "titles": titles,
        })
        print(f"[{label}]")
        print(f"  Query: {query}")
        print(f"  P@{k} = {precision} | scores = {scores}")
        print()
    return results


def save_csv(results: list[dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "label": r["label"],
            "query": r["query"],
            "k": r["k"],
            "retrieved_k": r["retrieved_k"],
            "precision_at_k": r["precision_at_k"],
            "scores": str(r["scores"]),
            "titles": " | ".join(r["titles"]),
        })

    df = pd.DataFrame(rows)
    mean_p = round(df["precision_at_k"].mean(), 4)

    print("--- RESULTADOS DE EVALUACIÓN ---")
    print(df[["label", "k", "precision_at_k"]].to_string(index=False))
    print(f"\nMean Precision@K: {mean_p}")

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    results = run()
    save_csv(results)
