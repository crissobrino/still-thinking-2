import time
import requests

def measure_guardrail_penalty(query, api_url):
    # 1. Medir con Guardrail (Asumiendo que eval_mode lo activa o es por defecto)
    start = time.perf_counter()
    res_with = requests.post(api_url, json={"query": query, "eval_mode": True}).json()
    end_with = time.perf_counter()
    latency_with = (end_with - start) * 1000 # a ms
    
    # 2. Medir sin Guardrail (Necesitas un flag en tu backend para saltarlo)
    start = time.perf_counter()
    res_without = requests.post(api_url, json={"query": query, "eval_mode": False}).json()
    end_without = time.perf_counter()
    latency_without = (end_without - start) * 1000
    
    penalty = latency_with - latency_without
    penalty_pct = (penalty / latency_without) * 100
    
    return {
        "baseline_ms": round(latency_without, 2),
        "with_guardrail_ms": round(latency_with, 2),
        "penalty_ms": round(penalty, 2),
        "penalty_pct": round(penalty_pct, 2)
    }

# penalty_data = measure_guardrail_penalty("What is BERT?", "http://localhost:8000/search")