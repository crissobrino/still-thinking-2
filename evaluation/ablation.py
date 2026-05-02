import seaborn as sns

configs = [
    {"name": "ANN Only", "pow": False, "rerank": False},
    {"name": "ANN + Reranker", "pow": False, "rerank": True},
    {"name": "ENN + Reranker (Gold)", "pow": True, "rerank": True}
]

ablation_data = []
# (Asumiendo que tienes una lista de queries 'test_queries')
for q in ["attention mechanism", "bert fine tuning"]:
    # Sacamos el Gold real primero para comparar
    gold_data, _ = call_rag(q, True, True)
    gold_ids = [a['id'] for a in gold_data.get('articles', [])]
    
    for cfg in configs:
        data, _ = call_rag(q, cfg['pow'], cfg['rerank'])
        test_ids = [a['id'] for a in data.get('articles', [])]
        ndcg = calculate_ndcg(gold_ids, test_ids)
        ablation_data.append({"Config": cfg['name'], "NDCG": ndcg, "Latency": data['metrics']['total_time']})

df_abl = pd.DataFrame(ablation_data)

# GRÁFICA A: Killer Chart (Calidad)
sns.barplot(x="Config", y="NDCG", data=df_abl, palette="viridis")
plt.title("Killer Chart: Impacto de la Arquitectura en la Calidad (NDCG@5)")
plt.show()

# GRÁFICA B: Latencia vs Calidad
sns.scatterplot(x="Latency", y="NDCG", hue="Config", s=100, data=df_abl)
plt.title("Trade-off: Latencia vs Calidad")
plt.xlabel("Tiempo de Respuesta (ms)")
plt.show()