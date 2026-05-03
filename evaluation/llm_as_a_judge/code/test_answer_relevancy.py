"""
Quick sanity-check: run the manual answer_relevancy judge on a set of
test cases loaded from a JSON file and print the REASON + SCORE for each.

JSON format:
[
    {
        "query": "transformer architectures for medical image segmentation",
        "answer": "- Research direction: ...",
        "expected": "high"   // optional: "high", "mid", "low"
    },
    ...
]

Run with:  python test_answer_relevancy.py
           python test_answer_relevancy.py path/to/custom_tests.json
"""

import os
import sys
import re
import json
from openai import OpenAI

HERE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, '..', '..', '..', 'backend-fastapi')
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, '.env'))

JUDGE_MODEL = "gpt-4o-mini"
_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPTS_DIR = os.path.join(HERE, '..', 'prompts')
with open(os.path.join(PROMPTS_DIR, 'answer_relevancy_system.txt')) as f:
    SYSTEM = f.read()
with open(os.path.join(PROMPTS_DIR, 'answer_relevancy_user.txt')) as f:
    USER_TEMPLATE = f.read()

# ── Load test cases ───────────────────────────────────────────────────────────
DEFAULT_JSON = os.path.join(HERE, 'synthetic_tests.json')
json_path    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON

if not os.path.exists(json_path):
    print(f"Test file not found: {json_path}")
    print("Create a JSON file with a list of {query, answer, expected} objects.")
    sys.exit(1)

with open(json_path) as f:
    test_cases = json.load(f)

print(f"Loaded {len(test_cases)} test case(s) from {json_path}")
print(f"Judge: {JUDGE_MODEL}\n")

# ── Score parser ──────────────────────────────────────────────────────────────
def parse_output(text: str) -> tuple[float | None, str]:
    reason_match = re.search(r'REASON:\s*(.+?)(?=Research direction:)', text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    # Compute from section scores if available
    section_scores = re.findall(r':\s*(\d+)/10', text)
    if len(section_scores) == 5:
        score = round(sum(int(s) for s in section_scores) / 50, 2)
        return score, reason or text

    # Fallback: parse SCORE line directly
    score_match = re.search(r'SCORE:\s*([0-9.]+(?:/50)?)', text)
    if score_match:
        raw = score_match.group(1)
        score = round(int(raw.split('/')[0]) / 50, 2) if '/' in raw else float(raw)
        return score, reason or text

    return None, reason or text

# ── Expected score ranges ─────────────────────────────────────────────────────
EXPECTED_RANGES = {
    "high": (0.75, 1.00),
    "mid":  (0.40, 0.74),
    "low":  (0.00, 0.39),
}

def check_expected(score: float, expected: str) -> str:
    if expected not in EXPECTED_RANGES:
        return ""
    lo, hi = EXPECTED_RANGES[expected]
    if lo <= score <= hi:
        return f"OK — within expected range for '{expected}' ({lo}–{hi})"
    return f"UNEXPECTED — expected '{expected}' ({lo}–{hi}) but got {score}"

# ── Run tests ─────────────────────────────────────────────────────────────────
for i, case in enumerate(test_cases, 1):
    query    = case["query"]
    answer   = case["answer"].strip()
    expected = case.get("expected", None)

    print(f"=== Test {i}/{len(test_cases)}: answer_relevancy ===")
    print(f"Query    : {query}")
    if expected:
        print(f"Expected : {expected}")
    print()

    user_msg = USER_TEMPLATE.format(q=query, ans=answer)

    try:
        response = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            timeout=60,
        )
        text  = response.choices[0].message.content.strip()
        score, reason = parse_output(text)

        print(f"REASON : {reason}")
        print(f"SCORE  : {score}")
        print()

        if score is None:
            print("WARNING: could not parse score — raw output below")
            print(text)
        elif expected:
            print(check_expected(score, expected))
        elif score >= 0.75:
            print("OK — high relevancy score")
        elif score >= 0.4:
            print("MID — moderate relevancy score")
        else:
            print("LOW — check if this is expected")

    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "─" * 60 + "\n")