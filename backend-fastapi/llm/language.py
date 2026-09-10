import langid

def detect_language(text: str) -> str:
    # Returns the ISO code (es, en, fr, de, etc.)
    lang, _ = langid.classify(text)
    return lang

def get_language_name(lang_code: str) -> str:
    # Maps the ISO code to the language name used in the LLM prompt
    names = {
        'es': 'Spanish',
        'en': 'English',
        'fr': 'French',
        'it': 'Italian',
        'de': 'German'
    }
    return names.get(lang_code, 'English')