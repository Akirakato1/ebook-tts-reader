from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import (
    AnnotationModelOutputError,
    JsonCompletionClient,
)
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
            return GlobalRegistryResult(characters=[dict(character) for character in characters])
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
        "Existing registry (compact identity only): "
        f"{json.dumps(known_characters, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Build a canonical character registry for audiobook voice casting.\n"
        "Existing registry is authoritative. Do not recreate characters already represented by role_id, "
        "display_name, aliases, profile_id, or person_id.\n"
        "Return only new characters or genuinely updated existing characters found in this chapter window; "
        "do not echo unchanged registry records.\n"
        "Do not produce sentence-level annotation or script rows.\n"
        "Merge aliases that clearly refer to the same person, such as first name, full name, title, or nickname.\n"
        "Create separate profiles only when the same person appears at a different life stage: child, teen, adult, or elder.\n"
        "Return JSON with exactly this shape: {\"characters\":[{\"name\":str,\"profile\":object,\"evidence\":list}]}.\n"
        "Each profile must include age_stage, gender, personality.\n"
        "Profile optional fields: profile_id, person_id, age, race_or_ethnicity, accent, timeline, aliases, same_person_as.\n"
        "Keep personality to short trait adjectives useful for voice casting.\n"
        "Use race_or_ethnicity and accent only when explicit or strongly text-grounded; otherwise null or omit.\n"
        "Evidence should be compact chapter references and short identity notes.\n\n"
        f"Chapter text:\n{rendered_chapters}\n\n"
        "Return JSON only. Do not wrap the JSON in Markdown code fences."
    )


def compact_registry_for_global_prompt(registry: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return compact

    for role_id, record in characters.items():
        if not isinstance(record, dict):
            continue
        compact_record = _compact_character_record(str(role_id), record)
        if compact_record:
            compact[str(role_id)] = compact_record
    return compact


def _compact_character_record(role_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    identity = _dict_value(record.get("identity_profile"))
    character_profile = _dict_value(record.get("character_profile"))
    compact: Dict[str, Any] = {}

    for key in ("role_id", "profile_id", "person_id", "display_name"):
        _set_if_present(compact, key, _first_present(record.get(key), role_id if key == "role_id" else None))

    for key in ("age", "age_stage", "gender", "race_or_ethnicity", "accent", "timeline"):
        _set_if_present(
            compact,
            key,
            _first_present(record.get(key), identity.get(key), character_profile.get(key)),
        )

    aliases = _compact_string_list(record.get("aliases"), max_items=12)
    if aliases:
        compact["aliases"] = aliases

    same_person_as = _compact_string_list(record.get("same_person_as"), max_items=8)
    if same_person_as:
        compact["same_person_as"] = same_person_as

    personality = _compact_string_list(
        _first_present(identity.get("personality"), character_profile.get("personality")),
        max_items=5,
    )
    if personality:
        compact["personality"] = personality

    return compact


def _dict_value(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _set_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        target[key] = value


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
