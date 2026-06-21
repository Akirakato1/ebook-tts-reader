from __future__ import annotations

import json
from typing import Dict, List

from ebook_tts_pipeline.domain import Sentence


SYSTEM_PROMPT = (
    "You label ebook sentences for audiobook generation. "
    "Return only valid JSON matching the requested compact schema."
)


def render_annotation_prompt(chapter: str, sentences: List[Sentence], registry: Dict) -> str:
    rendered_sentences = "\n".join(f"[{sentence.idx}] {sentence.text}" for sentence in sentences)
    allowed_indexes = [sentence.idx for sentence in sentences]
    known_characters = registry.get("characters", {})
    return (
        f"Known characters: {json.dumps(known_characters, ensure_ascii=False)}\n\n"
        f"Chapter: {chapter}\n\n"
        f"Chapter text:\n{rendered_sentences}\n\n"
        "Return JSON with these keys:\n"
        "- new_characters: list of {name, profile, voice}\n"
        "- Do not include Narrator in new_characters.\n"
        "- For each new character, profile must be an object.\n"
        "- For each new character, voice must be an object with non-empty string fields description and qwen_instruct.\n"
        "- roles: list of role names appearing in this window\n"
        '- Use exactly "Narrator" for narration, not "narrator" or another variant.\n'
        '- types: exactly ["narration", "dialogue", "thought"]\n'
        "- script: list of [role_idx, type_idx, sentence_idx]\n"
        f"- Allowed sentence_idx values: {json.dumps(allowed_indexes)}\n"
        f"- script must contain exactly {len(sentences)} rows, one for each allowed sentence_idx.\n"
        "Every sentence index in the input must appear exactly once.\n"
        "Do not wrap the JSON in Markdown code fences."
    )


def render_repair_prompt(original_prompt: str, invalid_output: Dict, errors: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed validation.\n"
        f"Validation errors: {errors}\n"
        f"Invalid JSON: {json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Return corrected JSON only."
    )
