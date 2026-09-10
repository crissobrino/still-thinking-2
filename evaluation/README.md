# Evaluation

Scripts and notebooks used to measure retrieval quality, generation quality, latency, and
guardrail behaviour for the RAG pipeline. Most scripts hit a running backend
(`http://localhost:8000`) or the ChromaDB store directly — start the backend first
(`uvicorn main:app --reload --port 8000` from `backend-fastapi/`) unless noted otherwise.

See [`EVALUATION_REPORT.md`](./EVALUATION_REPORT.md) for the write-up of results.

## Retrieval quality

| Script | What it measures |
| --- | --- |
| `recall.py` / `recall2.py` | Recall@K of ANN retrieval against a gold/exhaustive search baseline. |
| `main_eval.py` / `main_eval_mrr.py` | Overall retrieval metrics, including Mean Reciprocal Rank. |
| `ndcgg.py` | NDCG@K with bootstrapped confidence intervals. |
| `evaluate_precision.py` | Precision@K comparing ANN, ANN+reranking, and exact (ENN) search. |
| `compare_databases.py` | MiniLM vs Specter2 embedding backends, using MiniLM exact search as the gold standard. |
| `confusion_matrix.py` | Confusion matrix for relevant/irrelevant retrieval classification. |
| `ablation.py` | Ablation study over retrieval pipeline components. |
| `advanced_eval.py`, `evaluation_rag.py`, `evaluation_utils.py` | Shared helpers and additional pipeline-level metrics. |
| `generate_gold.py` | Builds the gold-standard relevance set used by the metrics above. |

## Generation quality (LLM-as-a-judge)

`llm_as_a_judge/` uses [RAGAS](https://github.com/explodinggradients/ragas) with an external LLM
judge (OpenAI) to score faithfulness, answer relevancy, and context precision/recall of the
generated answers. Requires a separate `OPENAI_API_KEY` and is not part of any automated test
suite — run manually.

- `code/evaluate_openai.py`, `code/evaluate_manual_openai.py` — run the RAGAS judge over the pipeline outputs.
- `code/evaluation_ragas.ipynb`, `code/evaluation_ragas_blue.ipynb` — exploratory/notebook runs.
- `code/plot_llm_eval.py` — plots the results in `results/`.
- `prompts/` — judge prompt templates.
- `results/small/`, `results/large/` — RAGAS output CSVs for the small (qwen3:8b) and large (qwen3:32b) models.

## Latency & language

- `time_models.py` — compares response latency between the small and large models.
- `language_benchmark.py`, `language_vis.py` — evaluate/plot translation and multilingual query handling.

## Guardrails

`guardrails/` evaluates the `should_refuse` guardrail: `guardrail_latency.py` measures its
overhead, `guardrail_check.ipynb` reviews refusal behaviour against `guardrail_queries*.json`.

## Running everything

`run_all_evals.py` runs the core retrieval-quality scripts end to end.
