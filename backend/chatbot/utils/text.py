import re


def is_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))


def clean_text(text: str, lang: str) -> str:
    text = text.strip()
    if lang == "fr":
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s']", "", text)
    elif lang == "ar":
        text = re.sub(r"[^\u0600-\u06FF0-9\s]", "", text)
    return text


def is_valid_text(text: str, lang: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    words = text.split()
    if len(words) < 2:
        return False
    if lang == "fr":
        valid = sum(1 for w in words if re.match(r"^[a-zA-ZÀ-ÿ']+$", w))
        return valid / len(words) > 0.6
    if lang == "ar":
        valid = sum(1 for w in words if re.match(r"^[\u0600-\u06FF]+$", w))
        return valid / len(words) > 0.6
    return True