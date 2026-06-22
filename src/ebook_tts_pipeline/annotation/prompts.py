from __future__ import annotations

import json
from typing import Any, Dict, List

from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt
from ebook_tts_pipeline.domain import Sentence


SYSTEM_PROMPT = (
    "You label ebook sentences for audiobook generation. "
    "Return only valid JSON matching the requested compact schema."
)


def render_annotation_prompt(
    chapter: str,
    sentences: List[Sentence],
    registry: Dict,
    lock_registry: bool = False,
) -> str:
    rendered_sentences = "\n".join(f"[{sentence.idx}] {sentence.text}" for sentence in sentences)
    allowed_indexes = [sentence.idx for sentence in sentences]
    known_characters = compact_registry_for_annotation_prompt(registry)
    character_schema = (
        "- local_speakers: list of {local_id, label, profile} for chapter-scoped temporary speakers "
        "not in the known character registry.\n"
        "- local_speakers are for unnamed one-off, background, or disposable speakers whose exact voice "
        "does not need consistency across chapters.\n"
        "- No approval is required for local_speakers; estimate a compact profile good enough for a fitting voice.\n"
    )
    return (
        f"Known characters: {json.dumps(known_characters, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"Chapter: {chapter}\n\n"
        f"Annotation units:\n{rendered_sentences}\n\n"
        "Return JSON with these keys:\n"
        f"{character_schema}"
        '- local_id must be stable within this chapter window, such as "tmp_001"; roles may use the local_id or label.\n'
        "- Every local_speakers item must include profile.\n"
        "- profile must be a JSON object, never null, never a string.\n"
        '- Example profile object: {"age_stage":"adult","gender":"female","personality":["guarded"]}.\n'
        "- For each new character, profile must be compact and only contain identity fields needed for future voice/profile decisions.\n"
        "- profile required fields: age_stage, gender, personality.\n"
        "- profile optional fields: profile_id, person_id, race_or_ethnicity, accent, occupation, aliases.\n"
        '- age_stage must be one of "child", "teen", "adult", "elder", or "unknown".\n'
        "- personality must be a short list of trait adjectives, such as shy, bright, charismatic, timid, guarded, hardened.\n"
        "- Use race_or_ethnicity and accent only when explicit or strongly text-grounded; otherwise use null or omit.\n"
        "- Do not put relationships, plot summary, backstory, grooming, abuse, or exploitation facts into voice-like fields.\n"
        "- Never frame grooming, exploitation, coercion, or child abuse as romance or consensual adult intimacy.\n"
        "- If the same underlying person appears at a different life stage, create a distinct profile_id such as callie_teen, callie_adult, trevor_child, or andrew_adult, and reuse the same person_id.\n"
        "- Do not append chapter, window, or sentence numbers to person_id or profile_id; use stable identity names like callie, callie_teen, or trevor_child.\n"
        "- Use the age-stage profile name in roles when needed to avoid ambiguity, such as Callie teen rather than Callie.\n"
        "- Known character summaries contain name and aliases; use one of those exact strings for roles.\n"
        "- If multiple known character summaries share the same name, use an alias with the age stage for that role.\n"
        "- roles: list of role names appearing in this window\n"
        '- Use exactly "Narrator" for narration, not "narrator" or another variant.\n'
        '- types: exactly ["narration", "dialogue", "thought"]\n'
        "- The chapter text has already been split into annotation units. A single source sentence can have multiple units.\n"
        "- If an input unit is narration around dialogue, such as said-tags or action beats outside quotes, label it Narrator/narration.\n"
        "- If an input unit is quoted external speech, label it as the speaking character/dialogue.\n"
        "- script: list of [role_idx, type_idx, unit_idx]\n"
        f"- Allowed unit_idx values: {json.dumps(allowed_indexes)}\n"
        f"- script must contain exactly {len(sentences)} rows, one for each allowed unit_idx.\n"
        "- Never emit multiple script rows for the same unit_idx.\n"
        "Every unit index in the input must appear exactly once.\n"
        "Do not wrap the JSON in Markdown code fences."
    )


def compact_registry_for_annotation_prompt(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return compact_registry_for_prompt(registry, include_aliases=True)


def render_repair_prompt(original_prompt: str, invalid_output: Dict, errors: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed validation.\n"
        f"Validation errors: {errors}\n"
        f"Invalid JSON: {json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Return corrected JSON only."
    )
