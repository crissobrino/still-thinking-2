import os
import sys
import json
import time
import re

# ── Path setup ────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, 'chroma_db'))

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

# ── Config ────────────────────────────────────────────────────────────────────
# One metric per run — avoids RAGAS batching multiple metrics together
METRIC = "faithfulness"   # faithfulness | answer_relevancy | context_recall

JUDGE_MODEL = "gpt-4o-mini"

# Short sleep between samples — OpenAI handles rate limits via max_retries
INTER_CALL_SLEEP = 5

# Set True to re-run the pipeline and overwrite the cache
FORCE_REFRESH = True

# Set True to use CrossEncoder reranking (fetch 20, rerank, keep top-5)
USE_RERANKER = False

# Set True to use brute-force exact search (ENN) instead of HNSW/ANN
USE_ENN = False

# Set True to use SPECTER2 DB instead of base all-MiniLM DB (mutually exclusive with USE_ENN)
USE_SPECTER = True

_openai_key = os.environ["OPENAI_API_KEY"]

# ── RAGAS judge setup ─────────────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate
from ragas.run_config import RunConfig
from datasets import Dataset


class _ResponseCapture(BaseCallbackHandler):
    """Collects all LLM responses during a RAGAS evaluate() call."""
    def __init__(self):
        self.responses: list[str] = []

    def reset(self):
        self.responses = []

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                text = g.text if hasattr(g, "text") else getattr(g.message, "content", "")
                if text:
                    self.responses.append(text.strip())

    def summary(self) -> str:
        return " | ".join(self.responses) if self.responses else ""


_capture = _ResponseCapture()

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=_openai_key,
        timeout=300,
        max_retries=3,
        callbacks=[_capture],
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

RUN_CFG = RunConfig(
    timeout=300,
    max_retries=3,
    max_wait=120,
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
QUERY_TYPES   = {q["query"]: q["type"] for q in _query_data}

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
    _query_data_full = json.load(open(os.path.join(HERE, '..', '..', 'test_queries.json')))["queries"]
    TEST_QUERIES  = [q["query"] for q in _query_data_full]
    GROUND_TRUTHS = {q["query"]: q["ground_truth"] for q in _query_data_full}

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

# ── Evaluate one sample at a time ─────────────────────────────────────────────
print(f"Evaluating [{METRIC}] sample-by-sample — judge: {JUDGE_MODEL}\n")

import pandas as pd

out_path = os.path.join(HERE, '..', 'results',
    f"automated_results_openai_{JUDGE_MODEL.replace(':', '-')}_{METRIC}.csv")

rows = []
n    = len(collected["question"])
t_total_start = time.perf_counter()

for i in range(n):
    q   = collected["question"][i]
    tag = QUERY_TYPES.get(q, "unknown")
    print(f"[{i+1}/{n}] [{tag}] {q[:70]}")

    # Expand the question to reflect the full task the system performs.
    # A bare topic like "transformers for medical imaging" doesn't match the answer's
    # structure (comparisons, differences, novelty). RAGAS reverse-question generation
    # needs the full intent to score answer_relevancy correctly.
    full_question = (
        f"Given the research idea '{q}', compare it with retrieved academic articles: "
        f"identify what each article covers, highlight key differences between the idea "
        f"and the existing works, and assess possible novelty."
    )
    single = Dataset.from_dict({
        "question":     [full_question],
        # The user query is prepended as a context so the faithfulness judge can verify
        # comparative statements ("the user's idea differs from..."). Without it,
        # RAGAS penalizes valid comparative reasoning as hallucination.
        "contexts":     [[f"User research query: {q}"] + collected["contexts"][i]],
        "answer":       [collected["answer"][i]],
        "ground_truth": [collected["ground_truth"][i]],
    })

    score, reason = None, ""
    for attempt in range(3):
        _capture.reset()
        try:
            result = evaluate(
                dataset=single,
                metrics=[METRIC_MAP[METRIC]],
                embeddings=evaluator_embeddings,
                run_config=RUN_CFG,
            )
            score  = result.to_pandas()[METRIC].iloc[0]
            reason = _capture.summary()
            print(f"  {METRIC}: {score:.3f}")
            break
        except Exception as e:
            if attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  [retry {attempt + 1}/2] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                reason = f"ERROR: {e}"
                print(f"  ERROR (giving up): {e}")

    rows.append({
        "query": q, "type": tag,
        "answer": collected["answer"][i],
        "ground_truth": collected["ground_truth"][i],
        METRIC: score,
        f"{METRIC}_reason": reason,
    })
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if i < n - 1:
        time.sleep(INTER_CALL_SLEEP)

# ── Aggregate ─────────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - t_total_start
df   = pd.DataFrame(rows)
vals = df[METRIC].dropna()
print(f"\n--- AGGREGATE RESULTS ---")
print(f"  {METRIC}: {vals.mean():.3f}  (n={len(vals)}/{n})")
print(f"  Total time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
print(f"\nSaved to {out_path}")
