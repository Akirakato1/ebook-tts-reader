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
        "- new_characters: list of {name, profile}\n"
        "- Do not include Narrator in new_characters.\n"
        "- Every new_characters item must include profile.\n"
        "- profile must be a JSON object, never null, never a string.\n"
        '- Example profile object: {"age_stage":"adult","gender":"female","personality":["guarded"]}.\n'
        "- For each new character, profile must be compact and only contain identity fields needed for future voice/profile decisions.\n"
        "- profile required fields: age_stage, gender, personality.\n"
        "- profile optional fields: profile_id, person_id, age, race_or_ethnicity, accent, timeline, aliases, same_person_as, narrative_notes.\n"
        '- age_stage must be one of "child", "teen", "adult", "elder", or "unknown".\n'
        "- personality must be a short list of trait adjectives, such as shy, bright, charismatic, timid, guarded, hardened.\n"
        "- Use race_or_ethnicity and accent only when explicit or strongly text-grounded; otherwise use null or omit.\n"
        "- Do not put relationships, plot summary, backstory, grooming, abuse, or exploitation facts into voice-like fields.\n"
        "- Put relationship or abuse context only in narrative_notes when needed for disambiguation or safety.\n"
        "- Never frame grooming, exploitation, coercion, or child abuse as romance or consensual adult intimacy.\n"
        "- If the same underlying person appears at a different life stage, create a distinct profile_id such as callie_teen, callie_adult, trevor_child, or andrew_adult, and reuse the same person_id.\n"
        "- Do not append chapter, window, or sentence numbers to person_id or profile_id; use stable identity names like callie, callie_teen, or trevor_child.\n"
        "- Use the age-stage profile name in roles when needed to avoid ambiguity, such as Callie teen rather than Callie.\n"
        "- roles: list of role names appearing in this window\n"
        '- Use exactly "Narrator" for narration, not "narrator" or another variant.\n'
        '- types: exactly ["narration", "dialogue", "thought"]\n'
        "- script: list of [role_idx, type_idx, sentence_idx]\n"
        f"- Allowed sentence_idx values: {json.dumps(allowed_indexes)}\n"
        f"- script must contain exactly {len(sentences)} rows, one for each allowed sentence_idx.\n"
        "- Never emit multiple script rows for the same sentence_idx, even if one sentence contains multiple quoted speakers.\n"
        "- For mixed-speaker sentence records, choose the first or primary speaker for the whole sentence.\n"
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
