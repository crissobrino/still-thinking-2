import numpy as np

def get_ndcg_with_confidence(ndcg_scores, confidence=0.95, n_bootstraps=1000):
    """
    Calcula la media del NDCG y su intervalo de confianza.
    """
    bootstrapped_means = []
    for _ in range(n_bootstraps):
        # Creamos una muestra aleatoria con reemplazo del mismo tamaño
        sample = np.random.choice(ndcg_scores, size=len(ndcg_scores), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower_bound = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    mean_ndcg = np.mean(ndcg_scores)
    
    return mean_ndcg, lower_bound, upper_bound

# Ejemplo de uso:
# scores = [0.8, 0.75, 0.9, 0.6, 0.85] # Tus NDCG de cada query
# mean, low, high = get_ndcg_with_confidence(scores)
# print(f"NDCG: {mean:.3f} (CI 95%: [{low:.3f}, {high:.3f}])")