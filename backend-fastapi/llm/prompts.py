from pathlib import Path

PROMPT_PATH = Path("prompts/comparison_prompt.txt")


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_comparison_prompt(query: str, context: str, language: str) -> str:
    template = load_prompt_template()

    try:
        return template.format(
            query=query,
            context=context,
            language=language
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in prompt template: {e}")