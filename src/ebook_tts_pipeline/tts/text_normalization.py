from __future__ import annotations


def normalize_tts_text(text: str) -> str:
    return " ".join(text.replace("#", " hashtag ").split())
