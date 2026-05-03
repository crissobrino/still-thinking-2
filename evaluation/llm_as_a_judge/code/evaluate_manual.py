import os
import sys
import json
import re
import requests as _req

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
# Pipeline generator — reuses pipeline_cache.json from evaluate_automated.py
_PIPELINE_MODEL = "llama3.1:8b"

# Judge model — qwen3:32b with think disabled via native Ollama API
_JUDGE_MODEL = "qwen3:32b"

# Set True to re-run the pipeline and overwrite the cache
FORCE_REFRESH = False

_ollama_key = os.environ["OLLAMA_API_KEY"]
_ollama_url = os.getenv("OLLAMA_URL", "https://yiyuan.tsc.uc3m.es")

# ── Pre-flight connectivity check ─────────────────────────────────────────────
def _check_connection():
    try:
        r = _req.get(
            f"{_ollama_url}/v1/models",
            headers={"X-API-KEY": _ollama_key, "Authorization": f"Bearer {_ollama_key}"},
            timeout=15,
        )
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"Server OK — models: {models}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        sys.exit(1)

_check_connection()

# ── Test queries ──────────────────────────────────────────────────────────────
_query_data   = json.load(open(os.path.join(HERE, '..', '..', 'test_queries.json')))["queries"]
TEST_QUERIES  = [q["query"]        for q in _query_data]
QUERY_TYPES   = {q["query"]: q["type"]         for q in _query_data}
GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data}

# ── Pipeline — reuses cache from evaluate_automated.py ───────────────────────
CACHE_PATH = os.path.join(HERE, '..', 'results', 'pipeline_cache.json')

if not FORCE_REFRESH and os.path.exists(CACHE_PATH):
    print(f"Loading pipeline results from cache ({_PIPELINE_MODEL})...\n")
    collected = json.load(open(CACHE_PATH))
else:
    from scripts.search_chroma import search
    from llm.client import UC3MClient
    from llm.prompts import build_comparison_prompt, build_system_prompt
    from llm.guardrails import should_refuse
    from llm.language import detect_language, get_language_name
    from deep_translator import GoogleTranslator

    os.environ["OLLAMA_MODEL"] = _PIPELINE_MODEL
    llm_client = UC3MClient()

    print(f"Running pipeline... [generator: {_PIPELINE_MODEL}]\n")
    collected = {"question": [], "contexts": [], "answer": [], "ground_truth": []}
    refused   = []

    for query in TEST_QUERIES:
        print(f"  → {query[:70]}")
        lang_code = detect_language(query)
        lang_name = get_language_name(lang_code)
        query_en  = (
            GoogleTranslator(source='auto', target='en').translate(query)
            if lang_code != 'en' else query
        )
        results = search(query_en, 5)
        scores  = [1 - r['distance'] for r in results]
        if should_refuse(results, scores):
            print("    [REFUSED]")
            refused.append(query)
            continue
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

if not collected["question"]:
    print("No results to evaluate.")
    sys.exit(0)

# ── Judge: one call per metric, only the relevant inputs ──────────────────────
# Each metric sees only what it needs — prevents cross-metric contamination.
#
# faithfulness:      context + answer    (does answer stay within context?)
# answer_relevancy:  question + answer   (does answer address the question?)
# context_precision: question + contexts (are retrieved passages relevant?)
# context_recall:    contexts + ground_truth (does context cover ground truth?)

def _ctx_block(contexts):
    return "\n\n".join(f"[Passage {i+1}]\n{c}" for i, c in enumerate(contexts))

_METRIC_PROMPTS = {
    "faithfulness": lambda q, ctxs, ans, gt: (
        "You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.\n\n"
        "FAITHFULNESS measures whether the system's answer is grounded in the provided context. "
        "It does not matter whether the answer is generally correct — only whether each claim "
        "in the answer can be directly traced back to the context passages. "
        "An answer that introduces facts from outside the context, even true ones, scores lower. "
        "An answer that only restates what the context says scores higher.",
        f"CONTEXT:\n{_ctx_block(ctxs)}\n\n"
        f"ANSWER:\n{ans}\n\n"
        "Given the context and the answer above, score the faithfulness of the answer.\n"
        "0.0 = no claim in the answer is supported by the context.\n"
        "1.0 = every claim in the answer is directly supported by the context.\n"
        "Reply in exactly this format:\nSCORE: <number between 0 and 1>\nREASON: <two sentences explaining the score>",
    ),
    "answer_relevancy": lambda q, ctxs, ans, gt: (
        "You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.\n\n"
        "ANSWER RELEVANCY measures how well the answer addresses the user's original question. "
        "A relevant answer directly responds to what was asked — it does not go off-topic, "
        "does not answer a different question, and does not give a generic response that could "
        "apply to any query. The quality or correctness of the answer is not assessed here, "
        "only whether it targets the right question.",
        f"QUESTION:\n{q}\n\n"
        f"ANSWER:\n{ans}\n\n"
        "Given the question and the answer above, score the answer relevancy.\n"
        "0.0 = the answer completely ignores or misses the question.\n"
        "1.0 = the answer directly and fully addresses the question.\n"
        "Reply in exactly this format:\nSCORE: <number between 0 and 1>\nREASON: <two sentences explaining the score>",
    ),
    "context_precision": lambda q, ctxs, ans, gt: (
        "You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.\n\n"
        "CONTEXT PRECISION measures the quality of the retrieval step: what fraction of the "
        "retrieved passages are actually useful for answering the question. "
        "A retrieval system that returns many off-topic passages scores lower, even if a few "
        "relevant ones are included. The goal is signal-to-noise in the retrieved context.",
        f"QUESTION:\n{q}\n\n"
        f"RETRIEVED PASSAGES:\n{_ctx_block(ctxs)}\n\n"
        "Given the question and the retrieved passages above, score the context precision.\n"
        "0.0 = none of the passages are relevant to the question.\n"
        "1.0 = all passages are directly relevant and useful for answering the question.\n"
        "Reply in exactly this format:\nSCORE: <number between 0 and 1>\nREASON: <two sentences explaining the score>",
    ),
    "context_recall": lambda q, ctxs, ans, gt: (
        "You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.\n\n"
        "CONTEXT RECALL measures how much of the information needed to answer the question — "
        "as captured in the ground truth — is actually present in the retrieved passages. "
        "A high score means the retrieval system found the right information. "
        "A low score means important information from the ground truth is missing from the context, "
        "which would make it impossible for any LLM to produce a complete answer.",
        f"RETRIEVED PASSAGES:\n{_ctx_block(ctxs)}\n\n"
        f"GROUND TRUTH ANSWER:\n{gt}\n\n"
        "Given the retrieved passages and the ground truth above, score the context recall.\n"
        "0.0 = none of the ground truth information is present in the retrieved passages.\n"
        "1.0 = all of the ground truth information can be found in the retrieved passages.\n"
        "Reply in exactly this format:\nSCORE: <number between 0 and 1>\nREASON: <two sentences explaining the score>",
    ),
}

def _judge(system: str, user: str) -> tuple[float | None, str]:
    payload = {
        "model": _JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "think": False},
    }
    r = _req.post(
        f"{_ollama_url}/api/chat",
        headers={"X-API-KEY": _ollama_key},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    score_match  = re.search(r'SCORE:\s*(1\.0*|0\.\d+|[01])', text)
    reason_match = re.search(r'REASON:\s*(.+)', text, re.DOTALL)
    score  = float(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else text
    return score, reason

# ── Evaluate ──────────────────────────────────────────────────────────────────
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
print(f"Evaluating all metrics [manual mode] using judge: {_JUDGE_MODEL}...\n")

rows = []
for i, q in enumerate(collected["question"]):
    tag   = QUERY_TYPES.get(q, "unknown")
    ctxs  = collected["contexts"][i]
    ans   = collected["answer"][i]
    gt    = collected["ground_truth"][i]

    print(f"[{i+1}/{len(collected['question'])}] [{tag}] {q[:70]}")
    scores  = {}
    reasons = {}
    for metric in METRICS:
        system, user = _METRIC_PROMPTS[metric](q, ctxs, ans, gt)
        try:
            score, reason = _judge(system, user)
            scores[metric]  = score
            reasons[metric] = reason
            print(f"  {metric}: {score}")
            print(f"    {reason}")
        except Exception as e:
            scores[metric]  = None
            reasons[metric] = f"ERROR: {e}"
            print(f"  {metric}: ERROR — {e}")

    rows.append({"query": q, "type": tag,
                 **scores,
                 **{f"{m}_reason": reasons[m] for m in METRICS}})

# ── Aggregate ─────────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(rows)
print("\n--- AGGREGATE RESULTS ---")
for m in METRICS:
    vals = df[m].dropna()
    print(f"  {m}: {vals.mean():.3f}  (n={len(vals)})")

out_path = os.path.join(HERE, '..', 'results', f"automated_results_manual_{_JUDGE_MODEL.replace(':', '-')}.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
