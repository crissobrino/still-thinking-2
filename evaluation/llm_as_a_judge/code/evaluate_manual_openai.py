import os
import sys
import json
import re
import time
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
METRIC      = "answer_relevancy"   # currently only answer_relevancy is defined
JUDGE_MODEL = "gpt-4o-mini"
INTER_CALL_SLEEP = 5
FORCE_REFRESH    = False
USE_RERANKER     = True
USE_ENN          = False
USE_SPECTER      = False

_openai_key = os.environ["OPENAI_API_KEY"]
_client     = OpenAI(api_key=_openai_key)

# ── Load prompts from disk ────────────────────────────────────────────────────
PROMPTS_DIR = os.path.join(HERE, '..', 'prompts')

def _load_prompt(metric: str, role: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{metric}_{role}.txt")
    with open(path) as f:
        return f.read()

SYSTEM_PROMPTS = {
    "answer_relevancy": _load_prompt("answer_relevancy", "system"),
}
USER_PROMPT_TEMPLATES = {
    "answer_relevancy": _load_prompt("answer_relevancy", "user"),
}

if METRIC not in SYSTEM_PROMPTS:
    print(f"No prompt defined for '{METRIC}'. Add {METRIC}_system.txt and {METRIC}_user.txt to prompts/")
    sys.exit(1)

# ── Test queries ──────────────────────────────────────────────────────────────
_query_data   = json.load(open(os.path.join(HERE, '..', '..', 'test_queries.json')))["queries"]
QUERY_TYPES   = {q["query"]: q["type"]         for q in _query_data}
GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data}

# ── Pipeline cache ────────────────────────────────────────────────────────────
_suffix = ("_specter" if USE_SPECTER else ("_enn" if USE_ENN else "")) + ("_reranked" if USE_RERANKER else "")
CACHE_PATH = os.path.join(HERE, '..', 'results', f'pipeline_cache_v2{_suffix}.json')

if FORCE_REFRESH:
    if USE_SPECTER:
        from specter.search_chroma import search as _search_fn
    elif USE_ENN:
        from scripts.search_chroma import search_enn_formatted as _search_fn
    else:
        from scripts.search_chroma import search as _search_fn
    from llm.client import UC3MClient
    from llm.prompts import build_comparison_prompt, build_system_prompt
    from llm.guardrails import should_refuse
    from llm.language import detect_language, get_language_name
    from deep_translator import GoogleTranslator

    if USE_RERANKER:
        import torch
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            default_activation_function=torch.nn.Sigmoid(),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    os.environ["OLLAMA_MODEL"] = "qwen3:8b"
    llm_client = UC3MClient()

    print(f"Running pipeline... [generator: qwen3:8b, db: {'specter' if USE_SPECTER else 'base'}, search: {'ENN' if USE_ENN else 'ANN'}, reranker: {USE_RERANKER}]\n")
    collected = {"question": [], "contexts": [], "answer": [], "ground_truth": []}
    refused   = []
    TEST_QUERIES  = [q["query"] for q in _query_data]

    for query in TEST_QUERIES:
        print(f"  → {query[:70]}")
        lang_code = detect_language(query)
        lang_name = get_language_name(lang_code)
        query_en  = (
            GoogleTranslator(source='auto', target='en').translate(query)
            if lang_code != 'en' else query
        )
        results = _search_fn(query_en, 20 if USE_RERANKER else 5)
        scores  = [1 - r['distance'] for r in results]
        if should_refuse(results, scores):
            print("    [REFUSED]")
            refused.append(query)
            continue
        if USE_RERANKER:
            pairs = [[query_en, r['document']] for r in results]
            rerank_scores = _reranker.predict(pairs)
            results = [r for r, s in sorted(zip(results, rerank_scores), key=lambda x: x[1], reverse=True)][:5]
        context = "".join(
            f"\n[Article {i}]\nTitle: {r['title']}\nAbstract: {r['document']}\n"
            for i, r in enumerate(results, 1)
        )
        answer = llm_client.chat(
            user_prompt=build_comparison_prompt(query, context, lang_name),
            system_prompt=build_system_prompt(lang_name),
        )
        collected["question"].append(query)
        collected["contexts"].append([r['document'] for r in results])
        collected["answer"].append(answer)
        collected["ground_truth"].append(GROUND_TRUTHS[query])
        print("    [OK]")

    json.dump(collected, open(CACHE_PATH, "w"))
    print(f"\nCollected {len(collected['question'])}, refused {len(refused)}\n")
elif os.path.exists(CACHE_PATH):
    print(f"Loading pipeline results from cache...\n")
    collected = json.load(open(CACHE_PATH))
else:
    print("No pipeline cache found. Set FORCE_REFRESH = True to generate it.")
    sys.exit(1)

if not collected["question"]:
    print("No results to evaluate.")
    sys.exit(0)

# Strip <think>...</think> blocks from pipeline answers before evaluation
collected["answer"] = [
    re.sub(r'<think>.*?</think>', '', a, flags=re.DOTALL).strip()
    for a in collected["answer"]
]

# ── Judge function ────────────────────────────────────────────────────────────
def _judge(system: str, user: str, retries: int = 3) -> tuple[float | None, str]:
    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0,
                timeout=60,
            )
            text = response.choices[0].message.content.strip()
            
            

            reason_match = re.search(r'REASON:\s*(.+?)(?=Research direction:)', text, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else text

            # Try to compute score from section scores
            section_scores = re.findall(r':\s*(\d+)/10', text)
            if len(section_scores) == 5:
                score = round(sum(int(s) for s in section_scores) / 50, 2)
            else:
                # Fallback: try to parse SCORE line directly
                score_match = re.search(r'SCORE:\s*([0-9.]+(?:/50)?)', text)
                if score_match:
                    raw = score_match.group(1)
                    score = round(int(raw.split('/')[0]) / 50, 2) if '/' in raw else float(raw)
                else:
                    score = None

            return score, reason
        except Exception as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    [retry {attempt + 1}/{retries - 1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise

# ── Evaluate ──────────────────────────────────────────────────────────────────
print(f"Evaluating [{METRIC}] — judge: {JUDGE_MODEL}\n")

import pandas as pd

out_path = os.path.join(HERE, '..', 'results',
    f"automated_results_manual_openai_{JUDGE_MODEL.replace(':', '-')}_{METRIC}.csv")

rows = []
n    = len(collected["question"])
t_start = time.perf_counter()

for i in range(n):
    q   = collected["question"][i]
    ans = collected["answer"][i]
    gt  = collected["ground_truth"][i]
    tag = QUERY_TYPES.get(q, "unknown")
    print(f"[{i+1}/{n}] [{tag}] {q[:70]}")

    system = SYSTEM_PROMPTS[METRIC]
    user   = USER_PROMPT_TEMPLATES[METRIC].format(q=q, ans=ans)

    score, reason = None, ""
    try:
        score, reason = _judge(system, user)
        print(f"  {METRIC}: {score}")
    except Exception as e:
        reason = f"ERROR: {e}"
        print(f"  ERROR: {e}")

    rows.append({
        "query": q, "type": tag,
        "answer": ans, "ground_truth": gt,
        METRIC: score, f"{METRIC}_reason": reason,
    })
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if i < n - 1:
        time.sleep(INTER_CALL_SLEEP)

# ── Aggregate ─────────────────────────────────────────────────────────────────
df   = pd.DataFrame(rows)
vals = df[METRIC].dropna()
print(f"\n--- AGGREGATE RESULTS ---")
print(f"  {METRIC}: {vals.mean():.3f}  (n={len(vals)}/{n})")
print(f"  Total time: {time.perf_counter() - t_start:.1f}s")
print(f"\nSaved to {out_path}")
