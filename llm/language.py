from langdetect import detect

#Language detection

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
    except Exception:
        return "English"

    if lang == "es":
        return "Spanish"
    return "English"