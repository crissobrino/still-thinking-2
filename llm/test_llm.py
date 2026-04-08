from llm.client import UC3MClient
from llm.language import detect_language
from llm.prompts import build_comparison_prompt

query = "I am researching federated neural topic models."

context = """
[Article 1]
Title: Federated topic modeling
Abstract: This paper studies federated approaches to topic modeling using probabilistic models.

[Article 2]
Title: Federated non-negative matrix factorization for short text topic modeling
Abstract: This paper explores federated NMF methods for short-text topic discovery.
"""

language = detect_language(query)
prompt = build_comparison_prompt(query, context, language)

client = UC3MClient()
response = client.chat(
    user_prompt=prompt,
    system_prompt="You are a precise academic comparison assistant.",
)

print(response)