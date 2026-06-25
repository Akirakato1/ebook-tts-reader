from __future__ import annotations

from typing import Any, Dict

from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    default_narrator_voice_profile,
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
            "voice_profile": default_narrator_voice_profile(),
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
        voice_profile = build_compact_voice_profile(display_name, {"identity_profile": identity})
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


def functional_narrator_voice_record(profile: Dict[str, Any]) -> Dict[str, Any]:
    base = narrator_voice_record(profile)
    voice_profile = dict(base["voice_profile"])
    base_description = str(voice_profile.get("description") or "audiobook narrator")
    base_instruction = str(voice_profile.get("qwen_instruct") or base_description)
    return {
        "role_id": "functional_narrator",
        "display_name": "Functional Narrator",
        "identity_profile": dict(base["identity_profile"]),
        "voice_identity": dict(base["voice_identity"]),
        "voice_profile": {
            "description": (
                f"{base_description}; same narrator identity for quoted non-dialogue text, "
                "slightly higher pitch, flatter monotone delivery, crisp and restrained"
            ),
            "qwen_instruct": (
                f"{base_instruction}. Keep the same base narrator identity, but render quoted "
                "non-dialogue text with a slightly higher pitch, flatter monotone cadence, "
                "restrained emotion, and crisp articulation."
            ),
        },
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]
