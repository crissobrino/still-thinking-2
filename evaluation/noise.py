adversarial_queries = [
    ("Impact of Harry Potter on Albacete GDP", "Nonsense"),
    ("Quantum teleportation of sandwiches", "Nonsense"),
    ("Transformers for NLP", "Scientific"),
]

correct_rejections = 0
for q_text, q_type in adversarial_queries:
    data, is_blocked = call_rag(q_text)
    if q_type == "Nonsense" and is_blocked:
        correct_rejections += 1
    elif q_type == "Scientific" and not is_blocked:
        correct_rejections += 1

print(f"Efectividad del Guardrail: {(correct_rejections/len(adversarial_queries))*100}%")