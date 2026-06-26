from __future__ import annotations

from typing import Any, Dict

from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    voice_profile_hash,
)
from ebook_tts_pipeline.voice_identity import role_seed


DEFAULT_NARRATOR_IDENTITY = {
    "age_stage": "adult",
    "gender": "male",
    "personality": ["calm", "clear", "measured"],
    "race_or_ethnicity": None,
    "accent": None,
    "occupation": "audiobook narrator",
}


def default_narrator_profile(book_slug: str = "book") -> Dict[str, Any]:
    return normalize_narrator_profile(
        {
            "role_id": "narrator",
            "display_name": "Narrator",
            "identity_profile": dict(DEFAULT_NARRATOR_IDENTITY),
            "voice_identity": {
                "seed": role_seed(book_slug, "narrator"),
                "differentiators": ["calm baseline narrator timbre"],
            },
        },
        book_slug=book_slug,
    )


def narrator_profile_from_registry(registry: Dict[str, Any], book_slug: str = "book") -> Dict[str, Any]:
    narrator = dict(registry.get("narrator") or {})
    if not narrator:
        return default_narrator_profile(book_slug)
    return normalize_narrator_profile(narrator, book_slug=book_slug)


def normalize_narrator_profile(profile: Dict[str, Any], book_slug: str = "book") -> Dict[str, Any]:
    role_id = "narrator"
    display_name = str(profile.get("display_name") or "Narrator").strip() or "Narrator"
    identity = dict(profile.get("identity_profile") or {})
    for key, value in DEFAULT_NARRATOR_IDENTITY.items():
        identity.setdefault(key, value)
    identity["personality"] = _string_list(identity.get("personality"))
    voice_identity = dict(profile.get("voice_identity") or {})
    voice_identity.setdefault("seed", role_seed(book_slug, role_id))
    voice_identity.setdefault("differentiators", ["calm baseline narrator timbre"])
    voice_profile = dict(profile.get("voice_profile") or {})
    if not voice_profile.get("description") or not voice_profile.get("qwen_instruct"):
        voice_profile = build_strict_narrator_voice_profile(display_name, identity)
    return {
        "role_id": role_id,
        "display_name": display_name,
        "identity_profile": identity,
        "voice_identity": voice_identity,
        "voice_profile": voice_profile,
    }


def narrator_profile_hash(profile: Dict[str, Any]) -> str:
    return voice_profile_hash(narrator_voice_record(profile))


def narrator_voice_record(profile: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_narrator_profile(profile)
    return {
        "role_id": "narrator",
        "display_name": normalized["display_name"],
        "identity_profile": dict(normalized["identity_profile"]),
        "voice_identity": dict(normalized["voice_identity"]),
        "voice_profile": dict(normalized["voice_profile"]),
    }


def build_strict_narrator_voice_profile(display_name: str, identity: Dict[str, Any]) -> Dict[str, str]:
    voice_profile = build_compact_voice_profile(display_name, {"identity_profile": identity})
    age_stage = str(identity.get("age_stage") or "adult").replace("_", " ").strip() or "adult"
    gender = str(identity.get("gender") or "unknown").strip() or "unknown"
    accent = _nullable_string(identity.get("accent"))
    race_or_ethnicity = _nullable_string(identity.get("race_or_ethnicity"))
    identity_phrase = " ".join(part for part in [age_stage, gender] if part and part != "unknown").strip()
    if not identity_phrase:
        identity_phrase = "adult narrator"

    description_parts = [str(voice_profile.get("description") or "").strip()]
    if accent:
        description_parts.append(f"strictly consistent {accent} pronunciation")
    description_parts.append("stable calm audiobook narrator timbre with no accent drift")

    instruction_parts = [str(voice_profile.get("qwen_instruct") or "").strip().rstrip(".")]
    if race_or_ethnicity:
        instruction_parts.append(
            f"Voice identity metadata: {race_or_ethnicity}; Do not infer a regional accent from race or ethnicity; "
            "accent is controlled only by the selected accent field"
        )
    if accent:
        instruction_parts.append(_strict_accent_instruction(accent))
    instruction_parts.append(
        f"Maintain the exact same {identity_phrase} calm audiobook narrator timbre across every sentence; "
        "No accent drift, no pitch drift, no character voice switching, no regional pronunciation changes, "
        "and no performance style change"
    )
    return {
        "description": "; ".join(part for part in description_parts if part),
        "qwen_instruct": ". ".join(part for part in instruction_parts if part) + ".",
    }


def narrator_summary(profile: Dict[str, Any]) -> str:
    normalized = normalize_narrator_profile(profile)
    identity = normalized["identity_profile"]
    parts = [
        str(identity.get("age_stage") or "").replace("_", " "),
        str(identity.get("gender") or ""),
        str(identity.get("accent") or "").replace("_", " "),
    ]
    compact = " ".join(part for part in parts if part and part != "unknown").strip()
    return f"{normalized['display_name']}: {compact or 'custom narrator'}"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strict_accent_instruction(accent: str) -> str:
    normalized = accent.strip().lower()
    if normalized == "general american":
        return (
            "Use consistent General American English pronunciation. Do not use British, Irish, Australian, "
            "Southern US, New York, or other regional accent features. Keep vowel shapes, rhythm, consonant "
            "articulation, and prosody stable across every generated sentence"
        )
    return (
        f"Use consistent {accent} pronunciation only. Do not switch to British, Irish, Australian, "
        "or unrelated regional accent features. Keep vowel shapes, rhythm, consonant articulation, "
        "and prosody stable across every generated sentence"
    )
