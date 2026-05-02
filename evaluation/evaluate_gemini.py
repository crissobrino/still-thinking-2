import os
import sys
import json
import requests as _req

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
# Metric to evaluate in this run:
#   "faithfulness" | "answer_relevancy" | "context_precision" | "context_recall"
METRIC = "faithfulness"

# Set True to re-run the pipeline and overwrite the cache
FORCE_REFRESH = False

# ── Backend imports ───────────────────────────────────────────────────────────
from scripts.search_chroma import search
from llm.client import UC3MClient
from llm.prompts import build_comparison_prompt, build_system_prompt
from llm.guardrails import should_refuse
from llm.language import detect_language, get_language_name
from deep_translator import GoogleTranslator

_PIPELINE_MODEL = "llama3.1:8b"
_JUDGE_MODEL    = "gemini-2.0-flash-lite"
_out_file       = f"automated_results_gemini_{METRIC}.csv"

_ollama_key = os.environ["OLLAMA_API_KEY"]
_ollama_url = os.getenv("OLLAMA_URL", "https://yiyuan.tsc.uc3m.es")
_google_key = os.environ["GOOGLE_API_KEY"]

# ── Pre-flight check — verify Gemini key works ────────────────────────────────
def _check_gemini():
    try:
        r = _req.get(
            f"https://generativelanguage.googleapis.com/v1beta/models/{_JUDGE_MODEL}",
            params={"key": _google_key},
            timeout=10,
        )
        r.raise_for_status()
        print(f"Gemini judge OK — model: {r.json().get('displayName', _JUDGE_MODEL)}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach Gemini API: {e}")
        sys.exit(1)

_check_gemini()

# ── RAGAS judge setup (Gemini 2.0 Flash-Lite) ─────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=_JUDGE_MODEL,
        google_api_key=_google_key,
        temperature=0.0,
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# ── Test queries ──────────────────────────────────────────────────────────────
_query_data   = json.load(open(os.path.join(HERE, "test_queries.json")))["queries"]
TEST_QUERIES  = [q["query"]        for q in _query_data]
QUERY_TYPES   = {q["query"]: q["type"]         for q in _query_data}
GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data}

# ── Pipeline — shares cache with evaluate_automated.py (same pipeline model) ──
CACHE_PATH = os.path.join(HERE, "pipeline_cache.json")

if not FORCE_REFRESH and os.path.exists(CACHE_PATH):
    print(f"Loading pipeline results from cache ({_PIPELINE_MODEL})...\n")
    collected = json.load(open(CACHE_PATH))
    refused   = []
else:
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

print(f"Evaluating with RAGAS [{METRIC}] using judge: {_JUDGE_MODEL}...")
dataset = Dataset.from_dict(collected)
results = evaluate(
    dataset=dataset,
    metrics=[METRIC_MAP[METRIC]],
    embeddings=evaluator_embeddings,
    run_config=RunConfig(max_workers=1, timeout=120, max_retries=3, max_wait=60),
)

print("\n--- RESULTS ---")
df = results.to_pandas()
for _, row in df.iterrows():
    q   = row['user_input']
    tag = QUERY_TYPES.get(q, "unknown")
    print(f"\n[{tag}] {q[:80]}")
    print(f"  {METRIC}: {row.get(METRIC, 'n/a')}")
print(f"\nAggregate {METRIC}: {results}")

out_path = os.path.join(HERE, _out_file)
results.to_pandas().to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
