"""
Measures and compares response latency between small (qwen3:8b) and large (qwen3:32b).
Run from repo root: python evaluation/time_models.py
"""
import os, sys, time
import ollama

HERE = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, '..', '.env'))

PROMPT = "Compare transformers and RNNs for sequence modelling in two sentences."

key = os.environ["OLLAMA_API_KEY"]
URL = os.getenv("OLLAMA_URL", "https://yiyuan.tsc.uc3m.es")

MODELS = {
    f"small ({os.getenv('OLLAMA_MODEL_FAST', 'qwen3:8b')})":    {"key": key, "model": os.getenv("OLLAMA_MODEL_FAST",    "qwen3:8b")},
    f"large ({os.getenv('OLLAMA_MODEL_QUALITY', 'qwen3:32b')})": {"key": key, "model": os.getenv("OLLAMA_MODEL_QUALITY", "qwen3:32b")},
}

for name, cfg in MODELS.items():
    client = ollama.Client(host=URL, headers={"X-API-KEY": cfg["key"]}, timeout=600)
    print(f"\n{name} — sending request...")
    t0 = time.perf_counter()
    chunks = client.chat(
        model=cfg["model"],
        messages=[{"role": "user", "content": PROMPT}],
        options={"temperature": 0.0, "think": False},
        stream=True,
    )
    response = "".join(c["message"]["content"] for c in chunks)
    elapsed = time.perf_counter() - t0
    print(f"  Time   : {elapsed:.2f}s")
    print(f"  Tokens : ~{len(response.split())} words")
    print(f"  Answer : {response[:150]}...")
