from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from ebook_tts_pipeline.domain import AnnotationResult, SentenceArtifact, SentenceUnit
from ebook_tts_pipeline.registry import normalize_name


QUOTE_OPENERS = ('"', "\u201c")
SPEECH_TAG_RE = re.compile(
    r"\b(said|asked|ordered|told|called|shouted|whispered|muttered|replied|answered|"
    r"yelled|cried|begged|warned|sang)\b",
    re.IGNORECASE,
)
PRONOUN_TAG_RE = re.compile(
    r"\b(?P<pronoun>he|she|they)\s+"
    r"(?:(?:still|then|quietly|softly|finally|again|just)\s+)?"
    r"(said|asked|ordered|told|called|shouted|whispered|muttered|replied|answered|"
    r"yelled|cried|begged|warned|sang)\b",
    re.IGNORECASE,
)


def normalize_mixed_dialogue_units(
    annotation: AnnotationResult,
    artifact: SentenceArtifact,
    registry: Dict[str, Any],
) -> AnnotationResult:
    """Correct deterministic quote/action labels after the model pass.

    The model still chooses speakers for hard dialogue, but quote boundaries are
    structural. In a source sentence split into quoted and outside-quote units,
    outside fragments are narration, while quoted fragments are dialogue.
    """

    unit_by_idx = {unit.idx: unit for unit in artifact.annotation_units}
    grouped_units: Dict[int, List[SentenceUnit]] = defaultdict(list)
    for unit in artifact.annotation_units:
        grouped_units[unit.sentence_idx].append(unit)

    roles = list(annotation.roles)
    types = list(annotation.types)
    narrator_idx = _ensure_item(roles, "Narrator")
    narration_idx = _ensure_item(types, "narration")
    dialogue_idx = _ensure_item(types, "dialogue")

    role_profiles = _role_profiles(annotation, registry)
    rows = {unit_idx: (role_idx, type_idx) for role_idx, type_idx, unit_idx in annotation.script}

    for sentence_units in grouped_units.values():
        quote_unit_ids = {unit.idx for unit in sentence_units if _is_quote_unit(unit)}
        if not quote_unit_ids or len(sentence_units) <= 1:
            continue

        inferred_quote_roles: Dict[int, int] = {}
        for index, unit in enumerate(sentence_units):
            if unit.idx in quote_unit_ids:
                continue
            tag_role = _role_from_speech_tag(
                unit.text,
                roles,
                role_profiles,
                previous_units=_previous_units(artifact.annotation_units, unit.idx),
            )
            if tag_role is None:
                continue
            previous_quote = _nearest_quote(sentence_units[:index], quote_unit_ids, reverse=True)
            next_quote = _nearest_quote(sentence_units[index + 1 :], quote_unit_ids, reverse=False)
            target = next_quote if _starts_with_pronoun_or_name_tag(unit.text, roles, role_profiles) else previous_quote
            if target is None:
                target = next_quote or previous_quote
            if target is not None:
                inferred_quote_roles[target.idx] = tag_role

        for unit in sentence_units:
            if unit.idx not in rows:
                continue
            role_idx, type_idx = rows[unit.idx]
            if unit.idx in quote_unit_ids:
                rows[unit.idx] = (inferred_quote_roles.get(unit.idx, role_idx), dialogue_idx)
            else:
                rows[unit.idx] = (narrator_idx, narration_idx)

    normalized_script: List[Tuple[int, int, int]] = []
    for _, _, unit_idx in annotation.script:
        if unit_idx not in unit_by_idx:
            normalized_script.append(next(row for row in annotation.script if row[2] == unit_idx))
            continue
        role_idx, type_idx = rows[unit_idx]
        normalized_script.append((role_idx, type_idx, unit_idx))

    return replace(annotation, roles=roles, types=types, script=normalized_script)


def _ensure_item(items: List[str], value: str) -> int:
    if value in items:
        return items.index(value)
    items.append(value)
    return len(items) - 1


def _is_quote_unit(unit: SentenceUnit) -> bool:
    return unit.text.strip().startswith(QUOTE_OPENERS)


def _nearest_quote(
    units: List[SentenceUnit],
    quote_unit_ids: set[int],
    reverse: bool,
) -> Optional[SentenceUnit]:
    candidates = reversed(units) if reverse else units
    for unit in candidates:
        if unit.idx in quote_unit_ids:
            return unit
    return None


def _previous_units(units: List[SentenceUnit], unit_idx: int, limit: int = 8) -> List[SentenceUnit]:
    previous = [unit for unit in units if unit.idx < unit_idx]
    return previous[-limit:]


def _starts_with_pronoun_or_name_tag(
    text: str,
    roles: List[str],
    role_profiles: Dict[str, Dict[str, Any]],
) -> bool:
    if PRONOUN_TAG_RE.search(text):
        return True
    if not SPEECH_TAG_RE.search(text):
        return False
    normalized_text = normalize_name(text)
    for role in roles:
        if role == "Narrator":
            continue
        for name in role_profiles.get(role, {}).get("names", []):
            if name and name in normalized_text:
                return True
    return False


def _role_from_speech_tag(
    text: str,
    roles: List[str],
    role_profiles: Dict[str, Dict[str, Any]],
    previous_units: List[SentenceUnit],
) -> Optional[int]:
    if not SPEECH_TAG_RE.search(text):
        return None

    normalized_text = normalize_name(text)
    for index, role in enumerate(roles):
        if role == "Narrator":
            continue
        for name in role_profiles.get(role, {}).get("names", []):
            if name and name in normalized_text:
                return index

    pronoun_match = PRONOUN_TAG_RE.search(text)
    if not pronoun_match:
        return None
    gender = _gender_for_pronoun(pronoun_match.group("pronoun"))
    if not gender:
        return None
    return _recent_role_for_gender(gender, roles, role_profiles, previous_units)


def _recent_role_for_gender(
    gender: str,
    roles: List[str],
    role_profiles: Dict[str, Dict[str, Any]],
    previous_units: List[SentenceUnit],
) -> Optional[int]:
    for unit in reversed(previous_units):
        normalized_text = normalize_name(unit.text)
        matches: List[int] = []
        for index, role in enumerate(roles):
            if role == "Narrator":
                continue
            profile = role_profiles.get(role, {})
            if profile.get("gender") != gender:
                continue
            if any(name and name in normalized_text for name in profile.get("names", [])):
                matches.append(index)
        if len(matches) == 1:
            return matches[0]
    return None


def _gender_for_pronoun(pronoun: str) -> str:
    normalized = pronoun.lower()
    if normalized == "he":
        return "male"
    if normalized == "she":
        return "female"
    return ""


def _role_profiles(annotation: AnnotationResult, registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for role in annotation.roles:
        profiles[role] = {"names": {normalize_name(role)}, "gender": ""}

    for character in registry.get("characters", {}).values():
        gender = _identity_gender(character)
        names = _character_names(character)
        for role in annotation.roles:
            if normalize_name(role) in names:
                profiles[role] = {"names": names | {normalize_name(role)}, "gender": gender}

    for speaker in annotation.local_speakers:
        local_id = str(speaker.get("local_id", ""))
        label = str(speaker.get("label", ""))
        profile = dict(speaker.get("profile", {})) if isinstance(speaker.get("profile"), dict) else {}
        names = {normalize_name(name) for name in (local_id, label) if name}
        names.update(_name_tokens(label))
        gender = str(profile.get("gender", "")).lower()
        for role in annotation.roles:
            if normalize_name(role) in names:
                profiles[role] = {"names": names | {normalize_name(role)}, "gender": gender}

    return profiles


def _identity_gender(record: Dict[str, Any]) -> str:
    identity = record.get("identity_profile", {})
    if not isinstance(identity, dict):
        identity = record.get("character_profile", {})
    if not isinstance(identity, dict):
        return ""
    return str(identity.get("gender", "")).lower()


def _character_names(character: Dict[str, Any]) -> set[str]:
    raw_names = {
        str(character.get("role_id", "")),
        str(character.get("role_id", "")).replace("_", " "),
        str(character.get("profile_id", "")),
        str(character.get("profile_id", "")).replace("_", " "),
        str(character.get("display_name", "")),
    }
    raw_names.update(str(alias) for alias in character.get("aliases", []))
    names = {normalize_name(name) for name in raw_names if name}
    for raw_name in raw_names:
        names.update(_name_tokens(raw_name))
    return names


def _name_tokens(name: str) -> set[str]:
    ignored = {"adult", "child", "teen", "elder", "unknown", "default", "internal"}
    tokens = {
        token.strip(".,;:!?")
        for token in name.replace("_", " ").split()
    }
    return {
        normalize_name(token)
        for token in tokens
        if len(normalize_name(token)) > 2 and token.lower() not in ignored
    }
