import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

API_URL = "http://localhost:8000/search"

def run_advanced_evaluation():
    print("🧪 Ejecutando Pruebas Avanzadas...")
    
    # 1. Matriz de Confusión (Ruido)
    test_queries = [
        ("How to boil an egg in quantum space?", 0),
        ("Who won the magic match in Harry Potter?", 0),
        ("Transformer self-attention mechanism explained", 1),
        ("BERT architecture for NLP", 1)
    ]
    y_true, y_pred = [], []
    
    for q, label in test_queries:
        resp = requests.post(API_URL, json={"query": q}).json()
        # Si la respuesta es el mensaje de rechazo o no hay artículos
        blocked = "sorry" in resp['answer'].lower() or len(resp['articles']) == 0
        y_true.append(label)
        y_pred.append(1 if not blocked else 0)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Blocked", "Passed"])
    disp.plot(cmap="Blues")
    plt.savefig("evaluation/guardrail_matrix.png")

    # 2. Token Efficiency y Penalty (Media sobre 3 queries)
    efficiencies, penalties = [], []
    for q in ["Deep learning", "Attention mechanism", "Quantum computing"]:
        data = requests.post(API_URL, json={"query": q}).json()
        eff = (data['metrics']['answer_tokens'] / data['metrics']['context_tokens']) * 100
        efficiencies.append(eff)
        penalties.append(data['metrics']['guardrail_time'])

    with open("evaluation/advanced_stats.txt", "w") as f:
        f.write(f"Mean Token Efficiency: {sum(efficiencies)/3:.2f}%\n")
        f.write(f"Mean Guardrail Penalty: {sum(penalties)/3:.2f}ms\n")

    print("✅ Pruebas de robustez y eficiencia completadas.")

if __name__ == "__main__":
    run_advanced_evaluation()