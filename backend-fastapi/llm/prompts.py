from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def build_system_prompt(language: str) -> str:
    try:
        return _load("system_prompt.txt").format(language=language)
    except KeyError as e:
        raise ValueError(f"Missing placeholder in system_prompt.txt: {e}")


def build_comparison_prompt(query: str, context: str, language: str) -> str:
    try:
        return _load("comparison_prompt.txt").format(
            query=query,
            context=context,
            language=language,
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in comparison_prompt.txt: {e}")


def build_summarize_prompt(query: str, abstract: str, language: str) -> str:
    try:
        return _load("summarize_prompt.txt").format(
            query=query,
            abstract=abstract,
            language=language,
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in summarize_prompt.txt: {e}")