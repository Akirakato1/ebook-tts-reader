from __future__ import annotations

from typing import Any, Dict, List


def compact_registry_for_prompt(
    registry: Dict[str, Any],
    include_aliases: bool = False,
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return compact

    for role_id, record in characters.items():
        if not isinstance(record, dict):
            continue
        compact_record = _compact_character_record(str(role_id), record, include_aliases=include_aliases)
        if compact_record:
            compact.append(compact_record)
    return compact


def _compact_character_record(
    role_id: str,
    record: Dict[str, Any],
    include_aliases: bool,
) -> Dict[str, Any]:
    identity = _dict_value(record.get("identity_profile"))
    character_profile = _dict_value(record.get("character_profile"))
    name = str(_first_present(record.get("display_name"), role_id)).strip()
    age_stage = str(
        _first_present(record.get("age_stage"), identity.get("age_stage"), character_profile.get("age_stage"), "unknown")
    ).strip()
    gender = str(_first_present(identity.get("gender"), character_profile.get("gender"), "unknown")).strip()
    compact: Dict[str, Any] = {
        "name": name,
        "age_stage": age_stage,
        "gender": gender,
        "race_or_accent": _race_or_accent(identity, character_profile),
        "occupation": str(_first_present(identity.get("occupation"), character_profile.get("occupation"), "unknown")),
        "personality_type": _personality_type(identity, character_profile),
    }
    if include_aliases:
        compact["aliases"] = _compact_aliases(name, age_stage, record.get("aliases"))
    return compact


def _compact_aliases(name: str, age_stage: str, aliases: Any) -> List[str]:
    compact = _compact_string_list(aliases, max_items=8)
    if age_stage and age_stage != "unknown":
        compact.append(f"{name} {age_stage.replace('_', ' ')}")
    return _dedupe_preserving_order(compact)


def _race_or_accent(
    identity: Dict[str, Any],
    character_profile: Dict[str, Any],
) -> str:
    race_or_ethnicity = _first_present(identity.get("race_or_ethnicity"), character_profile.get("race_or_ethnicity"))
    accent = _first_present(identity.get("accent"), character_profile.get("accent"))
    parts: List[str] = []
    if race_or_ethnicity:
        parts.append(str(race_or_ethnicity))
    if accent:
        parts.append(f"{accent} accent")
    return "; ".join(parts) if parts else "unknown"


def _personality_type(identity: Dict[str, Any], character_profile: Dict[str, Any]) -> str:
    personality = _compact_string_list(
        _first_present(identity.get("personality"), character_profile.get("personality")),
        max_items=5,
    )
    return ", ".join(personality) if personality else "unknown"


def _dict_value(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


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


def _dedupe_preserving_order(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        normalized = "".join(ch for ch in value.lower() if ch.isalnum())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped
