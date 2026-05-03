import os
import sys
import json
import requests as _req

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
# Metric to evaluate in this run:
#   "faithfulness" | "answer_relevancy" | "context_precision" | "context_recall"
METRIC = "faithfulness"

# Judge model:
#   "full"   → qwen3:8b
#   "strong" → qwen3:32b
EVAL_MODE = "strong"

# Set True to re-run the pipeline and overwrite the cache
FORCE_REFRESH = False

# ── Backend imports ───────────────────────────────────────────────────────────
from scripts.search_chroma import search
from llm.client import UC3MClient
from llm.prompts import build_comparison_prompt, build_system_prompt
from llm.guardrails import should_refuse
from llm.language import detect_language, get_language_name
from deep_translator import GoogleTranslator

# ── RAGAS judge setup ─────────────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset

_judge_model = "qwen3:8b" if EVAL_MODE == "full" else "qwen3:32b"
_out_file    = f"automated_results_{EVAL_MODE}_{METRIC}.csv"

_ollama_key = os.environ["OLLAMA_API_KEY"]
_ollama_url = os.getenv("OLLAMA_URL", "https://yiyuan.tsc.uc3m.es")

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=_judge_model,
        api_key=_ollama_key,
        base_url=f"{_ollama_url}/v1",
        default_headers={"X-API-KEY": _ollama_key},
        model_kwargs={"extra_body": {"think": False}},
        timeout=300,
        max_retries=2,
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

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
        print(f"Judge server OK ({r.status_code}) — models: {models}")
    except Exception as e:
        print(f"ERROR: Cannot reach judge server: {e}")
        sys.exit(1)

_check_connection()

# ── Test queries ──────────────────────────────────────────────────────────────
_queries_path = os.path.join(HERE, '..', '..', 'test_queries.json')
_query_data   = json.load(open(_queries_path))["queries"]
TEST_QUERIES  = [q["query"]        for q in _query_data]
QUERY_TYPES   = {q["query"]: q["type"]         for q in _query_data}
GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data}

# ── Pipeline — cached to avoid re-running on every metric ─────────────────────
CACHE_PATH = os.path.join(HERE, '..', 'results', 'pipeline_cache.json')

if not FORCE_REFRESH and os.path.exists(CACHE_PATH):
    print("Loading pipeline results from cache...\n")
    collected = json.load(open(CACHE_PATH))
    refused   = []
else:
    os.environ["OLLAMA_MODEL"] = "llama3.1:8b"
    llm_client = UC3MClient()

    print("Running pipeline for all test queries... [generator: llama3.1:8b]\n")
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
            print("    [REFUSED — no relevant articles above threshold]")
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
    print(f"\nCollected {len(collected['question'])} results, refused {len(refused)}")
    print(f"Pipeline cache saved to {CACHE_PATH}\n")

if not collected["question"]:
    print("No results to evaluate.")
    sys.exit(0)

# ── RAGAS evaluation ──────────────────────────────────────────────────────────
METRIC_MAP = {
    "faithfulness":      Faithfulness(llm=evaluator_llm),
    "answer_relevancy":  AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    "context_precision": ContextPrecision(llm=evaluator_llm),
    "context_recall":    ContextRecall(llm=evaluator_llm),
}

if METRIC not in METRIC_MAP:
    print(f"Unknown metric '{METRIC}'. Choose from: {list(METRIC_MAP)}")
    sys.exit(1)

print(f"Evaluating with RAGAS [{METRIC}] using judge: {_judge_model}...")
# The user query is prepended as a context so the faithfulness judge can verify
# comparative statements ("the user's idea differs from..."). Without it,
# RAGAS penalizes valid comparative reasoning as hallucination.
collected["contexts"] = [
    [f"User research query: {q}"] + ctxs
    for q, ctxs in zip(collected["question"], collected["contexts"])
]
dataset = Dataset.from_dict(collected)
results = evaluate(
    dataset=dataset,
    metrics=[METRIC_MAP[METRIC]],
    embeddings=evaluator_embeddings,
    run_config=RunConfig(max_workers=1, timeout=600, max_retries=2, max_wait=120),
)

print("\n--- RESULTS ---")
df = results.to_pandas()
for _, row in df.iterrows():
    q   = row['user_input']
    tag = QUERY_TYPES.get(q, "unknown")
    print(f"\n[{tag}] {q[:80]}")
    print(f"  {METRIC}: {row.get(METRIC, 'n/a')}")
print(f"\nAggregate {METRIC}: {results}")

# ── Save to CSV ───────────────────────────────────────────────────────────────
out_path = os.path.join(HERE, '..', 'results', _out_file)
results.to_pandas().to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
