import langid

def detect_language(text: str) -> str:
    # Retorna el código ISO (es, en, fr, de, etc.)
    lang, _ = langid.classify(text)
    return lang

def get_language_name(lang_code: str) -> str:
    # Mapeo para el prompt del LLM
    names = {
        'es': 'Spanish',
        'en': 'English',
        'fr': 'French',
        'it': 'Italian',
        'de': 'German'
    }
    return names.get(lang_code, 'English')