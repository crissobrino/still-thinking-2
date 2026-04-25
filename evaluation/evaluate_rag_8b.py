import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall

# 1. Define your test dataset manually (or load from a CSV)
test_queries = [
    "I am researching implementations of federated neural topic models.",
    "What are the coherence metrics used in Latent Dirichlet Allocation?",
    "I am researching systems for assisting researchers."
]

# Ground truths are the ideal answers or the core concepts that MUST be in the answer
ground_truths = [
    ["There are articles on federated topic modeling but none specifically on neural topic models."],
    ["UMass, UCI (PMI), NPMI, and Cv coherence."],
    ["No specific articles found in the corpus."]
]

# 2. Collect the data from YOUR system
# (You would typically write a loop here that calls your search() function and LLM)
answers = []
contexts = []

for query in test_queries:
    # Simulate calling your backend functions:
    # retrieved_docs = search(query, k=5)
    # response = call_llama_uc3m(query, context)
    
    # For this example, let's assume we collected these from your system:
    answers.append("The retrieved articles focus on Bayesian models, but no neural implementations were found.")
    contexts.append([
        "Abstract of paper 1 about Federated topic modeling...", 
        "Abstract of paper 2 about Federated NMF..."
    ])

# 3. Format the data for Ragas
data = {
    "question": test_queries,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

# 4. Run Ragas Evaluation
print("Starting Ragas evaluation... This may take a few minutes.")

# Note: Ragas uses an LLM to evaluate your LLM. 
# By default it tries to use OpenAI. You can pass your custom UC3M Langchain wrapper here if needed.
results = evaluate(
    dataset = dataset, 
    metrics = [faithfulness, answer_relevance, context_precision, context_recall]
)

# 5. Print and save the results for your UC3M Report
df = results.to_pandas()
print(df)
df.to_csv("ragas_evaluation_results.csv", index=False)