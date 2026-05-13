import re
import unicodedata


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _normalize_apostrophes(s: str) -> str:
    return s.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")


def _is_latin_letter(ch: str) -> bool:
    if len(ch) != 1:
        return False
    try:
        return unicodedata.name(ch).startswith("LATIN ")
    except ValueError:
        return False


def clean_text(text: str, lang: str) -> str:
    """
    Strip noise while keeping real letters (incl. œ, accented capitals, etc.).
    The old FR regex [^a-zA-ZÀ-ÿ...] dropped U+0153 (œ) and many Latin letters.
    """
    text = unicodedata.normalize("NFKC", (text or "").strip())
    text = _normalize_apostrophes(text)
    if lang == "fr":
        out: list[str] = []
        for ch in text:
            if ch.isspace():
                out.append(" ")
            elif ch in "'-":
                out.append(ch)
            elif ch.isdigit():
                out.append(ch)
            elif _is_latin_letter(ch):
                out.append(ch)
            else:
                out.append(" ")
        text = re.sub(r"\s+", " ", "".join(out)).strip()
    elif lang == "ar":
        text = re.sub(r"[^\u0600-\u06FF0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def _fr_token_ok(word: str) -> bool:
    w = word.strip()
    if not w:
        return False
    if w.isdigit():
        return True
    core = [c for c in w if c not in "'-"]
    if not core:
        return False
    for c in w:
        if c in "'-":
            continue
        if c.isdigit():
            continue
        if not _is_latin_letter(c):
            return False
    return True


def _ar_token_ok(word: str) -> bool:
    return bool(re.match(r"^[\u0600-\u06FF0-9]+$", word))


def is_valid_text(text: str, lang: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    words = text.split()
    if len(words) < 2:
        return len(words[0]) >= 2
    if lang == "fr":
        ok = sum(1 for w in words if _fr_token_ok(w))
        return ok / len(words) >= 0.45
    if lang == "ar":
        ok = sum(1 for w in words if _ar_token_ok(w))
        return ok / len(words) >= 0.45
    return True
