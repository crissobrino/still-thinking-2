import requests
import json

def generate_synthetic_queries(seeds, n_per_seed=5):
    synthetic_dataset = []
    print(f"Generando {len(seeds) * n_per_seed} consultas...")

    for seed in seeds:
        # Prompt para el LLM (ajusta según tu modelo)
        prompt = f"Genera {n_per_seed} preguntas de investigación diferentes sobre '{seed}'. 
                  Incluye algunas cortas y otras muy técnicas. Solo devuelve la lista de preguntas."
        
        # Aquí llamas a tu backend o directamente a Ollama/OpenAI
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "llama3", "prompt": prompt, "stream": False})
        
        # Limpieza básica de la respuesta
        queries = response.json()['response'].split('\n')
        synthetic_dataset.extend([q.strip() for q in queries if len(q) > 10])
    
    return synthetic_dataset

# Ejemplo de uso
seeds = ["Quantum computing", "Bioinformatics", "Neuroscience", "Deep Learning"]
new_queries = generate_synthetic_queries(seeds)