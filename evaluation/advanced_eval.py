import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

API_URL = "http://localhost:8000/search"

def run_advanced_evaluation():
    print("🧪 Running advanced tests...")

    # 1. Confusion matrix (noise queries)
    #test_queries = [
    #    ("How to boil an egg in quantum space?", 0),
    #    ("Who won the magic match in Harry Potter?", 0),
    #    ("Transformer self-attention mechanism explained", 1),
    #    ("BERT architecture for NLP", 1)
    #]

    test_queries = [
        ("fine-tuning BERT for text classification tasks", 1),
        ("large language models for abstractive text summarization", 1),
        ("aprendizaje automático para clasificación de texto", 1),
        ("redes neuronales para procesamiento del lenguaje natural", 1),
        ("knowledge graph embeddings for link prediction", 1),
        ("sentiment analysis using deep learning", 1),
        ("deep learning", 1),
        ("quantum computing error correction", 0),
        ("history of the Roman Empire", 0),
        ("attention", 1),
        ("quiero cocinar pasta carbonara y necesito los ingredientes", 0),
        ("integrales sin IA", 0),
        ("manger une pomme de terre", 0),
        ("Aller en montagne le matin donne mal à la tête", 0),
    ]

    y_true, y_pred = [], []
    
    for q, label in test_queries:
        resp = requests.post(API_URL, json={"query": q}).json()
        # Blocked if the response is the refusal message or has no articles
        blocked = "sorry" in resp['answer'].lower() or len(resp['articles']) == 0
        y_true.append(label)
        y_pred.append(1 if not blocked else 0)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Blocked", "Passed"])
    disp.plot(cmap="Blues")
    plt.savefig("evaluation/guardrail_matrix.png")

    # 2. Token efficiency and guardrail penalty (mean over 3 queries)
    efficiencies, penalties = [], []
    for q in ["Deep learning", "Attention mechanism", "Quantum computing"]:
        data = requests.post(API_URL, json={"query": q}).json()
        metrics = data.get('metrics', {})

        # Safely extract tokens (defaults to 0 if missing)
        ans_tok = metrics.get('answer_tokens', 0)
        ctx_tok = metrics.get('context_tokens', 0)

        # Only compute efficiency when there's context to measure against
        if ctx_tok > 0:
            eff = (ans_tok / ctx_tok) * 100
            efficiencies.append(eff)
        else:
            print(f"⚠️ Warning: no tokens detected for '{q}'. (Likely blocked by the guardrail)")

        penalties.append(metrics.get('guardrail_time', 0))

    # Only save if we managed to compute at least one efficiency value
    mean_eff = sum(efficiencies)/len(efficiencies) if efficiencies else 0
    mean_pen = sum(penalties)/len(penalties) if penalties else 0

    with open("evaluation/advanced_stats.txt", "w") as f:
        f.write(f"Mean Token Efficiency: {mean_eff:.2f}%\n")
        f.write(f"Mean Guardrail Penalty: {mean_pen:.2f}ms\n")

    print("✅ Robustness and efficiency tests completed.")

if __name__ == "__main__":
    run_advanced_evaluation()