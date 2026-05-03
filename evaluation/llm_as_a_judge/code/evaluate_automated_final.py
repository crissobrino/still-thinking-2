import os
import sys
import json
import time
import requests as _req

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
# One metric per run — avoids RAGAS batching multiple metrics together
METRIC = "faithfulness"   # faithfulness | answer_relevancy | context_precision | context_recall

# Small, fast judge — different family from the llama3.1 pipeline to reduce self-eval bias
JUDGE_MODEL = "qwen3:8b"  # qwen3:8b | gemma2:9b

# Seconds to sleep between samples — prevents overwhelming the shared GPU server
INTER_CALL_SLEEP = 10

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

# ── RAGAS judge setup ─────────────────────────────────────────────────────────
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset


class _CleanOutputChatOpenAI(ChatOpenAI):
    """Strips <think>...</think> blocks before RAGAS parses the response."""

    @staticmethod
    def _strip(text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _clean(self, result: ChatResult) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=self._strip(g.message.content)
                                      if isinstance(g.message.content, str)
                                      else g.message.content),
                    generation_info=g.generation_info,
                )
                for g in result.generations
            ],
            llm_output=result.llm_output,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._clean(super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._clean(await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))


evaluator_llm = LangchainLLMWrapper(
    _CleanOutputChatOpenAI(
        model=JUDGE_MODEL,
        api_key=_ollama_key,
        base_url=f"{_ollama_url}/v1",
        default_headers={"X-API-KEY": _ollama_key},
        model_kwargs={"extra_body": {"think": False}},
        timeout=600,
        max_retries=3,
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Sequential, long timeout, retries — the main levers against hanging
RUN_CFG = RunConfig(
    max_workers=1,
    timeout=600,
    max_retries=3,
    max_wait=180,
)

METRIC_MAP = {
    "faithfulness":      Faithfulness(llm=evaluator_llm),
    "answer_relevancy":  AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    "context_precision": ContextPrecision(llm=evaluator_llm),
    "context_recall":    ContextRecall(llm=evaluator_llm),
}

if METRIC not in METRIC_MAP:
    print(f"Unknown metric '{METRIC}'. Choose from: {list(METRIC_MAP)}")
    sys.exit(1)

# ── Test queries ──────────────────────────────────────────────────────────────
_query_data   = json.load(open(os.path.join(HERE, '..', '..', 'test_queries.json')))["queries"]
TEST_QUERIES  = [q["query"]        for q in _query_data]
QUERY_TYPES   = {q["query"]: q["type"]         for q in _query_data}
GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data}

# ── Pipeline — reuses cache from evaluate_automated.py ───────────────────────
CACHE_PATH = os.path.join(HERE, '..', 'results', 'pipeline_cache.json')

if not FORCE_REFRESH and os.path.exists(CACHE_PATH):
    print(f"Loading pipeline results from cache...\n")
    collected = json.load(open(CACHE_PATH))
else:
    from scripts.search_chroma import search
    from llm.client import UC3MClient
    from llm.prompts import build_comparison_prompt, build_system_prompt
    from llm.guardrails import should_refuse
    from llm.language import detect_language, get_language_name
    from deep_translator import GoogleTranslator

    os.environ["OLLAMA_MODEL"] = "llama3.1:8b"
    llm_client = UC3MClient()

    print("Running pipeline... [generator: llama3.1:8b]\n")
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

# ── Evaluate one sample at a time ─────────────────────────────────────────────
# Passing the whole dataset at once lets RAGAS batch and parallelize internally
# even with max_workers=1. Evaluating sample-by-sample gives us full control:
# per-sample timeout, graceful skip on failure, and incremental saves.
print(f"Evaluating [{METRIC}] sample-by-sample — judge: {JUDGE_MODEL}\n")

import pandas as pd

out_path = os.path.join(HERE, '..', 'results',
    f"automated_results_final_{JUDGE_MODEL.replace(':', '-')}_{METRIC}.csv")

rows = []
n    = len(collected["question"])

for i in range(n):
    q   = collected["question"][i]
    tag = QUERY_TYPES.get(q, "unknown")
    print(f"[{i+1}/{n}] [{tag}] {q[:70]}")

    single = Dataset.from_dict({
        "question":     [q],
        # The user query is prepended as a context so the faithfulness judge can verify
        # comparative statements ("the user's idea differs from..."). Without it,
        # RAGAS penalizes valid comparative reasoning as hallucination.
        "contexts":     [[f"User research query: {q}"] + collected["contexts"][i]],
        "answer":       [collected["answer"][i]],
        "ground_truth": [collected["ground_truth"][i]],
    })

    score = None
    try:
        result = evaluate(
            dataset=single,
            metrics=[METRIC_MAP[METRIC]],
            embeddings=evaluator_embeddings,
            run_config=RUN_CFG,
        )
        score = result.to_pandas()[METRIC].iloc[0]
        print(f"  {METRIC}: {score:.3f}")
    except Exception as e:
        print(f"  ERROR (skipping): {e}")

    rows.append({"query": q, "type": tag, METRIC: score})

    # Save after every sample so a crash doesn't lose earlier results
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if i < n - 1:
        time.sleep(INTER_CALL_SLEEP)

# ── Aggregate ─────────────────────────────────────────────────────────────────
df   = pd.DataFrame(rows)
vals = df[METRIC].dropna()
print(f"\n--- AGGREGATE RESULTS ---")
print(f"  {METRIC}: {vals.mean():.3f}  (n={len(vals)}/{n})")
print(f"\nSaved to {out_path}")
