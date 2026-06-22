from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

from ebook_tts_pipeline.annotation.prompts import SYSTEM_PROMPT
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt
from ebook_tts_pipeline.registry import normalize_name


ALLOWED_QUOTE_TYPES = {"dialogue", "narrator_quote"}


class QuoteAttributionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class QuoteAttributionResult:
    roles: List[str]
    quotes: List[Tuple[int, int, str]]
    local_speakers: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuoteAttributionResult":
        return cls(
            roles=[str(role) for role in data["roles"]],
            local_speakers=list(data.get("local_speakers", [])),
            quotes=[
                (int(row[0]), int(row[1]), str(row[2]))
                for row in data["quotes"]
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "roles": self.roles,
            "quotes": [list(row) for row in self.quotes],
        }
        if self.local_speakers:
            payload["local_speakers"] = self.local_speakers
        return payload


def render_quote_attribution_prompt(
    chapter: str,
    extraction: QuoteExtraction,
    registry: Dict[str, Any],
) -> str:
    compact_registry = _compact_registry_with_role_ids(registry)
    quote_ids = [quote.quote_id for quote in extraction.quotes]
    return (
        "You are attributing quoted dialogue in a novel chapter for audiobook generation.\n\n"
        f"Chapter: {chapter}\n\n"
        "Global recurring characters. Existing registry roles are authoritative:\n"
        f"{json.dumps(compact_registry, ensure_ascii=False, indent=2)}\n\n"
        "Chapter text with marked quotes:\n"
        f"{extraction.to_marked_text()}\n\n"
        f"Quote IDs to attribute: {json.dumps(quote_ids)}\n\n"
        "Rules:\n"
        "- Attribute every marked quote exactly once.\n"
        "- Choose a global role_id when the speaker is a recurring registry character.\n"
        "- If the same person has multiple age stages in the registry, choose the active age-stage role_id.\n"
        "- If the speaker is not in the global registry and appears chapter-only, create a local speaker.\n"
        "- Do not create global registry characters in this output.\n"
        "- Do not label normal quoted dialogue as Narrator.\n"
        "- Use narrator_quote only when quote marks are not spoken dialogue, such as titles, quoted terms, or sarcasm.\n"
        "- Return JSON only. Do not include quote text or explanations.\n\n"
        "Output schema:\n"
        "{\n"
        '  "roles": ["role_id_or_local_id"],\n'
        '  "local_speakers": [\n'
        "    {\n"
        '      "local_id": "local_001",\n'
        '      "label": "short visible name",\n'
        '      "profile": {\n'
        '        "age_stage": "adult|child|teen|elder|unknown",\n'
        '        "gender": "female|male|nonbinary|unknown",\n'
        '        "race_or_ethnicity": null,\n'
        '        "accent": null,\n'
        '        "occupation": null,\n'
        '        "personality": ["short trait"]\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "quotes": [[1, 0, "dialogue"]]\n'
        "}\n"
    )


def validate_quote_attribution(
    result: QuoteAttributionResult,
    quote_indices: Iterable[int],
    known_role_ids: Set[str],
) -> None:
    errors: List[str] = []
    expected = set(int(index) for index in quote_indices)
    seen: Set[int] = set()
    duplicate: Set[int] = set()
    local_ids = _validate_local_speakers(result.local_speakers, errors)
    normalized_known = {normalize_name(role_id) for role_id in known_role_ids}

    for quote_idx, role_idx, quote_type in result.quotes:
        if quote_idx in seen:
            duplicate.add(quote_idx)
        seen.add(quote_idx)
        if role_idx < 0 or role_idx >= len(result.roles):
            errors.append(f"role index out of range for quote {quote_idx}: {role_idx}")
            continue
        if quote_type not in ALLOWED_QUOTE_TYPES:
            errors.append(f"invalid quote type for quote {quote_idx}: {quote_type}")
            continue
        role = result.roles[role_idx]
        normalized_role = normalize_name(role)
        if quote_type == "dialogue" and normalized_role == normalize_name("Narrator"):
            errors.append(f"Narrator cannot speak dialogue quote {quote_idx}")
        if (
            normalized_role != normalize_name("Narrator")
            and normalized_role not in normalized_known
            and normalized_role not in local_ids
        ):
            errors.append(f"local role missing profile for quote {quote_idx}: {role}")

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append(f"missing quote assignments: {missing}")
    if extra:
        errors.append(f"unknown quote assignments: {extra}")
    if duplicate:
        errors.append(f"duplicate quote assignments: {sorted(duplicate)}")

    if errors:
        raise QuoteAttributionValidationError("; ".join(errors))


class QuoteAttributionService:
    def __init__(self, client: Any) -> None:
        self.client = client

    def attribute_quotes(
        self,
        chapter: str,
        extraction: QuoteExtraction,
        registry: Dict[str, Any],
    ) -> QuoteAttributionResult:
        prompt = render_quote_attribution_prompt(chapter, extraction, registry)
        payload = self.client.complete_json(SYSTEM_PROMPT, prompt)
        result = QuoteAttributionResult.from_dict(payload)
        validate_quote_attribution(
            result,
            quote_indices=[quote.idx for quote in extraction.quotes],
            known_role_ids=set(_registry_role_ids(registry)),
        )
        return result


def _compact_registry_with_role_ids(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = compact_registry_for_prompt(registry, include_aliases=True)
    role_ids = list(_registry_role_ids(registry))
    for index, record in enumerate(compact):
        if index < len(role_ids):
            record["role_id"] = role_ids[index]
    return compact


def _registry_role_ids(registry: Dict[str, Any]) -> List[str]:
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return []
    return [str(record.get("role_id") or role_id) for role_id, record in characters.items() if isinstance(record, dict)]


def _validate_local_speakers(
    local_speakers: List[Dict[str, Any]],
    errors: List[str],
) -> Set[str]:
    local_ids: Set[str] = set()
    for speaker in local_speakers:
        local_id = str(speaker.get("local_id", "")).strip()
        profile = speaker.get("profile")
        if not local_id:
            errors.append("local speaker missing local_id")
            continue
        if not isinstance(profile, dict):
            errors.append(f"local speaker missing profile: {local_id}")
            continue
        local_ids.add(normalize_name(local_id))
    return local_ids
