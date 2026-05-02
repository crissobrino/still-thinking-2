import matplotlib.pyplot as plt
import pandas as pd

queries_multilang = [
    {"en": "Deep learning in medicine", "es": "Aprendizaje profundo en medicina", "fr": "Apprentissage profond en médecine"},
    {"en": "Quantum error correction", "es": "Corrección de errores cuánticos", "fr": "Correction d'erreurs quantiques"}
]

results = []
for q in queries_multilang:
    ids_by_lang = {}
    for lang, txt in q.items():
        data, _ = call_rag(txt)
        ids_by_lang[lang] = [a['id'] for a in data.get('articles', [])]
    
    # Intersección Jaccard (Similitud entre idiomas)
    intersection = set(ids_by_lang['en']) & set(ids_by_lang['es']) & set(ids_by_lang['fr'])
    overlap = len(intersection) / 5.0
    results.append(overlap)

# Gráfica
plt.bar([f"Q{i+1}" for i in range(len(results))], results, color='teal')
plt.title("Robustez de Traducción: Solapamiento de Artículos (%)")
plt.ylabel("Coeficiente de Similitud")
plt.ylim(0, 1)
plt.show()