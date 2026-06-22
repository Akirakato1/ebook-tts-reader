from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import (
    AnnotationModelOutputError,
    JsonCompletionClient,
)
from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt
from ebook_tts_pipeline.debug_logging import FailureLogger


GLOBAL_REGISTRY_SYSTEM_PROMPT = (
    "You build canonical ebook character registries for audiobook casting. "
    "Return only valid JSON matching the requested schema."
)


@dataclass(frozen=True)
class GlobalRegistryChapter:
    chapter: str
    title: str
    text: str


@dataclass(frozen=True)
class GlobalRegistryResult:
    characters: List[Dict[str, Any]]


class GlobalRegistryService:
    def __init__(
        self,
        client: JsonCompletionClient,
        failure_logger: Optional[FailureLogger] = None,
    ) -> None:
        self.client = client
        self.failure_logger = failure_logger

    def discover_characters(
        self,
        book_title: str,
        registry: Dict[str, Any],
        chapters: List[GlobalRegistryChapter],
    ) -> GlobalRegistryResult:
        prompt = render_global_registry_prompt(book_title, registry, chapters)
        try:
            payload = self.client.complete_json(GLOBAL_REGISTRY_SYSTEM_PROMPT, prompt)
            characters = payload.get("characters", [])
            if not isinstance(characters, list):
                raise AnnotationModelOutputError("Global registry JSON field 'characters' must be a list.")
            return GlobalRegistryResult(
                characters=[_normalize_global_character_delta(character) for character in characters]
            )
        except Exception as exc:
            if self.failure_logger is not None:
                self.failure_logger.write_failure(
                    "global_registry_error",
                    {
                        "book_title": book_title,
                        "chapters": [chapter.chapter for chapter in chapters],
                        "system_prompt": GLOBAL_REGISTRY_SYSTEM_PROMPT,
                        "user_prompt": prompt,
                        "raw_model_text": getattr(exc, "raw_text", None),
                    },
                    exc=exc,
                )
            raise


def render_global_registry_prompt(
    book_title: str,
    registry: Dict[str, Any],
    chapters: List[GlobalRegistryChapter],
) -> str:
    rendered_chapters = "\n\n".join(
        f"## {chapter.chapter}: {chapter.title}\n{chapter.text}" for chapter in chapters
    )
    known_characters = compact_registry_for_global_prompt(registry)
    return (
        f"Book title: {book_title}\n\n"
        "Existing character summaries: "
        f"{json.dumps(known_characters, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Build a canonical character registry for audiobook voice casting.\n"
        "Only include named, story-important characters expected to have dialogue or internal thoughts on page.\n"
        "Do not include people who are only mentioned, referenced, described, or part of backstory unless they "
        "speak or think on page.\n"
        "Do not include unnamed one-off speakers, crowd/background roles, purely functional scene roles, or "
        "animals/pets unless they have spoken or thought lines.\n"
        "Existing registry is authoritative. Do not recreate characters already represented in the summaries.\n"
        "Return new characters and existing-character updates only when this chapter window adds or corrects "
        "these key facts: name, age_stage, gender, race_or_ethnicity, accent, occupation, personality. "
        "Do not echo unchanged registry records.\n"
        "Do not produce sentence-level annotation or script rows.\n"
        "Merge aliases that clearly refer to the same person, such as first name, full name, title, or nickname.\n"
        "Create separate profiles only when the same person appears at a different life stage: child, teen, adult, or elder.\n"
        "Return JSON with exactly this shape: "
        "{\"characters\":[{\"name\":str,\"age_stage\":str,\"gender\":str,"
        "\"race_or_accent\":str,\"occupation\":str,\"personality_type\":str}]}.\n"
        "Return one row per character life-stage variant; use the same name with a different age_stage "
        "when a child, teen, adult, or elder version should have a distinct voice profile.\n"
        "Use age_stage values child, teen, adult, elder, or unknown.\n"
        "Use race_or_accent as a compact string such as 'Japanese; Tokyo accent', or 'unknown'.\n"
        "Keep personality to short trait adjectives useful for voice casting.\n"
        "Use race or accent facts only when explicit or strongly text-grounded; otherwise use 'unknown'.\n"
        f"Chapter text:\n{rendered_chapters}\n\n"
        "Return JSON only. Do not wrap the JSON in Markdown code fences."
    )


def compact_registry_for_global_prompt(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    return compact_registry_for_prompt(registry, include_aliases=False)


def _normalize_global_character_delta(character: Any) -> Dict[str, Any]:
    if not isinstance(character, dict):
        raise AnnotationModelOutputError("Global registry character entries must be objects.")
    name = str(character.get("name", "")).strip()
    if not name:
        raise AnnotationModelOutputError("Global registry character entries must include name.")
    profile = character.get("profile")
    if isinstance(profile, dict):
        return {"name": name, "profile": dict(profile)}

    race_or_ethnicity, accent = _parse_race_or_accent(character.get("race_or_accent"))
    normalized_profile: Dict[str, Any] = {
        "age_stage": _compact_unknown_string(character.get("age_stage")),
        "gender": _compact_unknown_string(character.get("gender")),
        "personality": _split_compact_list(character.get("personality_type")),
    }
    if race_or_ethnicity is not None:
        normalized_profile["race_or_ethnicity"] = race_or_ethnicity
    if accent is not None:
        normalized_profile["accent"] = accent
    occupation = _nullable_compact_string(character.get("occupation"))
    if occupation is not None:
        normalized_profile["occupation"] = occupation
    return {"name": name, "profile": normalized_profile}


def _parse_race_or_accent(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "none", "null", "n/a"}:
        return None, None

    race_or_ethnicity: Optional[str] = None
    accent: Optional[str] = None
    for part in [item.strip() for item in text.split(";") if item.strip()]:
        lowered = part.lower()
        if lowered in {"unknown", "none", "null", "n/a"}:
            continue
        if lowered.endswith(" accent"):
            accent = part[: -len(" accent")].strip() or None
        elif lowered == "accent":
            continue
        elif race_or_ethnicity is None:
            race_or_ethnicity = part
        else:
            race_or_ethnicity = f"{race_or_ethnicity}; {part}"
    return race_or_ethnicity, accent


def _compact_unknown_string(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return text if text and text not in {"none", "null", "n/a"} else "unknown"


def _nullable_compact_string(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return None if not text or text.lower() in {"unknown", "none", "null", "n/a"} else text


def _split_compact_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _compact_string_list(value, max_items=5)
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null", "n/a"}:
        return []
    return _compact_string_list(re.split(r"[,;]", text), max_items=5)


def _compact_string_list(value: Any, max_items: int) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    compact: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        normalized = "".join(ch for ch in text.lower() if ch.isalnum())
        if normalized in seen:
            continue
        seen.add(normalized)
        compact.append(text)
        if len(compact) >= max_items:
            break
    return compact
