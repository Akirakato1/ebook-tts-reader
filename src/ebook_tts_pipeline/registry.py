from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Set

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.voice_identity import (
    append_differentiators,
    choose_differentiators,
    role_seed,
)

DEPRECATED_CHARACTER_FIELDS = (
    "age",
    "timeline",
    "same_person_as",
    "character_profile",
    "narrative_notes",
    "first_seen",
    "global_evidence",
)


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot create role_id from empty name: {name!r}")
    return slug


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def voice_profile_hash(voice_record: Dict[str, Any]) -> str:
    profile = voice_record.get("voice_profile", {})
    signature = "\n".join(
        [
            str(profile.get("description", "")),
            str(profile.get("qwen_instruct", "")),
        ]
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def voice_variant_for_type(speech_type: str) -> str:
    return "internal" if speech_type == "thought" else "default"


def profile_id_for_character(name: str, profile: Dict[str, Any]) -> str:
    explicit = str(profile.get("profile_id", "")).strip()
    if explicit:
        return slugify_name(explicit)
    person_id = slugify_name(str(profile.get("person_id", "")).strip() or name)
    age_stage = slugify_name(str(profile.get("age_stage", "")).strip() or "unknown")
    if age_stage and age_stage != "unknown":
        return f"{person_id}_{age_stage}"
    return person_id


def ensure_character_voice_variants(book_slug: str, character: Dict[str, Any]) -> None:
    if "voice_variants" in character:
        return
    if "voice_profile" not in character:
        return

    role_id = str(character["role_id"])
    display_name = str(character["display_name"])
    base_voice = dict(character["voice_profile"])
    identity = dict(character.get("voice_identity", {}))
    seed = int(identity.get("seed", role_seed(book_slug, role_id)))
    differentiators = list(identity.get("differentiators", choose_differentiators(book_slug, role_id)))

    character["voice_variants"] = {
        "default": {
            "role_id": f"{role_id}_default",
            "display_name": f"{display_name}_default",
            "voice_identity": {"seed": seed, "differentiators": differentiators},
            "voice_profile": base_voice,
            "voice_config_path": None,
        },
        "internal": {
            "role_id": f"{role_id}_internal",
            "display_name": f"{display_name}_internal",
            "voice_identity": {"seed": seed, "differentiators": differentiators},
            "voice_profile": _internal_voice_profile(display_name, base_voice),
            "voice_config_path": None,
        },
    }


def resolve_effective_voice(
    registry: Dict[str, Any],
    role_name: str,
    speech_type: str,
) -> Dict[str, Any]:
    narrator = registry.get("narrator", {})
    narrator_names = {
        normalize_name(str(narrator.get("display_name", "Narrator"))),
        normalize_name(str(narrator.get("role_id", "narrator"))),
        normalize_name("Narrator"),
    }
    if normalize_name(role_name) in narrator_names:
        return {
            "character": None,
            "role": str(narrator.get("display_name", "Narrator")),
            "role_id": str(narrator.get("role_id", "narrator")),
            "voice_variant": None,
            "voice_record": narrator,
        }

    book_slug = str(registry.get("book", {}).get("slug", "book"))
    normalized = normalize_name(role_name)
    display_counts = Counter(
        normalize_name(str(character.get("display_name", "")))
        for character in registry.get("characters", {}).values()
        if character.get("display_name")
    )
    for character in registry.get("characters", {}).values():
        ensure_character_voice_variants(book_slug, character)
        include_display_name = display_counts[normalize_name(str(character.get("display_name", "")))] == 1
        direct_names = _character_lookup_names(character, include_display_name=include_display_name)
        variant_match = _matching_variant(character, normalized)
        if normalized in direct_names or variant_match:
            variant_key = variant_match or voice_variant_for_type(speech_type)
            return _effective_voice_for_character(character, variant_key, role_name)

    short_matches = [
        character
        for character in registry.get("characters", {}).values()
        if normalize_name(_first_display_token(str(character.get("display_name", "")))) == normalized
    ]
    if len(short_matches) == 1:
        character = short_matches[0]
        ensure_character_voice_variants(book_slug, character)
        return _effective_voice_for_character(character, voice_variant_for_type(speech_type), role_name)

    raise ValueError(f"No registry record exists for annotated role: {role_name}")


def _internal_voice_profile(display_name: str, base_voice: Dict[str, Any]) -> Dict[str, Any]:
    description = str(base_voice.get("description", "")).rstrip(". ")
    qwen_instruct = str(base_voice.get("qwen_instruct", "")).rstrip(". ")
    return {
        "description": (
            f"{description}; same {display_name} identity for internal monologue, "
            "closer, softer, reflective, less projected"
        ).strip("; "),
        "qwen_instruct": (
            f"{qwen_instruct}. Keep the same {display_name} speaker identity and timbre, "
            "but perform this as internal monologue: closer, softer, inward, reflective, "
            "and less projected. Do not whisper unless the text itself implies whispering."
        ).strip(),
    }


def _effective_voice_for_character(
    character: Dict[str, Any],
    variant_key: str,
    fallback_role: str,
) -> Dict[str, Any]:
    variant = character.get("voice_variants", {}).get(variant_key)
    if variant:
        return {
            "character": str(character.get("display_name", fallback_role)),
            "role": str(variant.get("display_name", fallback_role)),
            "role_id": str(variant.get("role_id", character.get("role_id", fallback_role))),
            "voice_variant": variant_key,
            "voice_record": variant,
        }
    return {
        "character": str(character.get("display_name", fallback_role)),
        "role": str(character.get("display_name", fallback_role)),
        "role_id": str(character.get("role_id", fallback_role)),
        "voice_variant": None,
        "voice_record": character,
    }


def _first_display_token(display_name: str) -> str:
    parts = display_name.split()
    if not parts:
        return ""
    token = parts[0].strip(".,;:!?")
    return token if len(normalize_name(token)) > 2 else ""


def _character_lookup_names(character: Dict[str, Any], include_display_name: bool = True) -> Set[str]:
    names = [
        str(character.get("role_id", "")),
        str(character.get("role_id", "")).replace("_", " "),
        str(character.get("profile_id", "")),
        str(character.get("profile_id", "")).replace("_", " "),
    ]
    if include_display_name:
        names.append(str(character.get("display_name", "")))
    names.extend(str(alias) for alias in character.get("aliases", []))
    return {normalize_name(name) for name in names if name}


def _lookup_names_for_collision(character: Dict[str, Any]) -> Set[str]:
    names = [
        str(character.get("role_id", "")),
        str(character.get("role_id", "")).replace("_", " "),
        str(character.get("profile_id", "")),
        str(character.get("profile_id", "")).replace("_", " "),
    ]
    names.extend(str(alias) for alias in character.get("aliases", []))
    return {normalize_name(name) for name in names if name}


def _matching_variant(character: Dict[str, Any], normalized_name: str) -> str:
    for variant_key, variant in character.get("voice_variants", {}).items():
        names = [
            str(variant.get("display_name", "")),
            str(variant.get("role_id", "")),
            str(variant.get("role_id", "")).replace("_", " "),
        ]
        if normalized_name in {normalize_name(name) for name in names if name}:
            return str(variant_key)
    return ""


class RegistryManager:
    def __init__(self, paths: BookPaths) -> None:
        self.paths = paths

    def initialize_if_missing(self, book_title: str, book_slug: str) -> None:
        if self.paths.registry.exists():
            return
        registry: Dict[str, Any] = {
            "book": {"title": book_title, "slug": book_slug},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_identity": {
                    "seed": role_seed(book_slug, "narrator"),
                    "differentiators": ["calm baseline narrator timbre"],
                },
                "voice_profile": {
                    "description": "calm literary narrator, clear pacing",
                    "qwen_instruct": "A calm literary narrator voice with clear pacing.",
                },
                "voice_config_path": None,
            },
            "characters": {},
        }
        write_json_atomic(self.paths.registry, registry)

    def load(self) -> Dict[str, Any]:
        return read_json(self.paths.registry)

    def save(self, registry: Dict[str, Any]) -> None:
        prune_deprecated_registry_fields(registry)
        write_json_atomic(self.paths.registry, registry)

    def known_names(self) -> Set[str]:
        registry = self.load()
        names = {"Narrator"}
        for character in registry.get("characters", {}).values():
            names.update(str(name) for name in _lookup_names_for_collision(character))
            names.update(str(alias) for alias in character.get("aliases", []))
        return names

    def add_new_characters(self, chapter: str, new_characters: List[Dict[str, Any]]) -> None:
        registry = self.load()
        normalized_known = {normalize_name(name) for name in self.known_names()}
        book_slug = str(registry["book"]["slug"])

        for character in new_characters:
            name = str(character["name"]).strip()
            profile = normalize_character_profile(name, character.get("profile", {}))
            role_id = str(profile["profile_id"])
            collision_names = {normalize_name(name), normalize_name(role_id), normalize_name(role_id.replace("_", " "))}
            collision_names.update(normalize_name(alias) for alias in profile["aliases"])
            collision = sorted(collision_names & normalized_known)
            if collision:
                raise ValueError(f"collides with existing character or alias: {role_id}")
            differentiators = choose_differentiators(book_slug, role_id)
            voice = build_compact_voice_profile(name, profile)
            voice["qwen_instruct"] = append_differentiators(
                str(voice["qwen_instruct"]),
                differentiators,
            )
            seed = role_seed(book_slug, role_id)
            registry["characters"][role_id] = {
                "role_id": role_id,
                "profile_id": role_id,
                "person_id": profile["person_id"],
                "display_name": name,
                "age_stage": profile["age_stage"],
                "aliases": profile["aliases"],
                "identity_profile": profile["identity_profile"],
                "voice_identity": {
                    "seed": seed,
                    "differentiators": differentiators,
                },
                "voice_variants": {
                    "default": {
                        "role_id": f"{role_id}_default",
                        "display_name": f"{name}_default",
                        "voice_identity": {"seed": seed, "differentiators": differentiators},
                        "voice_profile": voice,
                        "voice_config_path": None,
                    },
                    "internal": {
                        "role_id": f"{role_id}_internal",
                        "display_name": f"{name}_internal",
                        "voice_identity": {"seed": seed, "differentiators": differentiators},
                        "voice_profile": _internal_voice_profile(name, voice),
                        "voice_config_path": None,
                    },
                },
            }
            prune_deprecated_character_fields(registry["characters"][role_id])
            normalized_known.add(normalize_name(role_id))
            normalized_known.add(normalize_name(role_id.replace("_", " ")))
            normalized_known.update(normalize_name(alias) for alias in profile["aliases"])

        self.save(registry)

    def merge_global_characters(self, chapter: str, characters: List[Dict[str, Any]]) -> None:
        registry = self.load()
        book_slug = str(registry["book"]["slug"])

        for character in characters:
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            profile = normalize_character_profile(name, character.get("profile", {}))
            role_id = _find_matching_character_id(registry, name, profile)
            if role_id:
                _merge_character_record(
                    registry["characters"][role_id],
                    name=name,
                    profile=profile,
                    evidence=character.get("evidence", []),
                    book_slug=book_slug,
                )
                continue
            self.add_new_characters(
                chapter=chapter,
                new_characters=[{"name": name, "profile": character.get("profile", {})}],
            )
            registry = self.load()

        prune_deprecated_registry_fields(registry)
        self.save(registry)


def normalize_character_profile(name: str, raw_profile: Any) -> Dict[str, Any]:
    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    age_stage = str(profile.get("age_stage", "unknown")).strip().lower().replace(" ", "_") or "unknown"
    gender = str(profile.get("gender", "unknown")).strip().lower() or "unknown"
    person_id = slugify_name(str(profile.get("person_id", "")).strip() or name)
    profile_id = profile_id_for_character(name, {**profile, "person_id": person_id, "age_stage": age_stage})
    personality = _string_list(profile.get("personality"))
    identity_profile = {
        "age_stage": age_stage,
        "gender": gender,
        "personality": personality,
        "race_or_ethnicity": _nullable_string(
            profile.get("race_or_ethnicity", profile.get("race", profile.get("ethnicity")))
        ),
        "accent": _nullable_string(profile.get("accent")),
        "occupation": _nullable_string(profile.get("occupation", profile.get("job", profile.get("profession")))),
    }
    aliases = _string_list(profile.get("aliases"))
    if age_stage != "unknown":
        aliases.append(f"{name} {age_stage.replace('_', ' ')}")
    return {
        "profile_id": profile_id,
        "person_id": person_id,
        "age_stage": age_stage,
        "aliases": _dedupe_preserving_order(aliases),
        "identity_profile": identity_profile,
    }


def _find_matching_character_id(
    registry: Dict[str, Any],
    name: str,
    profile: Dict[str, Any],
) -> str:
    candidates = {
        normalize_name(name),
        normalize_name(str(profile.get("profile_id", ""))),
        normalize_name(str(profile.get("profile_id", "")).replace("_", " ")),
        normalize_name(str(profile.get("person_id", ""))),
    }
    candidates.update(normalize_name(alias) for alias in profile.get("aliases", []))
    candidates.discard("")
    for role_id, record in registry.get("characters", {}).items():
        names = _character_lookup_names(record)
        names.add(normalize_name(str(record.get("person_id", ""))))
        if candidates & names:
            return str(role_id)
    return ""


def _merge_character_record(
    record: Dict[str, Any],
    name: str,
    profile: Dict[str, Any],
    evidence: Any,
    book_slug: str,
) -> None:
    existing_identity = dict(record.get("identity_profile", record.get("character_profile", {})))
    incoming_identity = dict(profile.get("identity_profile", {}))
    personality = _dedupe_preserving_order(
        _string_list(existing_identity.get("personality")) + _string_list(incoming_identity.get("personality"))
    )
    merged_identity = dict(existing_identity)
    for key in ("age_stage", "gender", "race_or_ethnicity", "accent", "occupation"):
        incoming_value = incoming_identity.get(key)
        existing_value = merged_identity.get(key)
        if existing_value in (None, "", "unknown") and incoming_value not in (None, "", "unknown"):
            merged_identity[key] = incoming_value
    merged_identity["personality"] = personality

    aliases = _string_list(record.get("aliases")) + _string_list(profile.get("aliases"))
    if normalize_name(name) != normalize_name(str(record.get("display_name", ""))):
        aliases.append(name)
    record["aliases"] = _dedupe_preserving_order(aliases)
    record["identity_profile"] = merged_identity
    prune_deprecated_character_fields(record)
    _refresh_record_voice_profiles(book_slug, record)


def prune_deprecated_character_fields(character: Dict[str, Any]) -> None:
    for key in DEPRECATED_CHARACTER_FIELDS:
        character.pop(key, None)
    identity = character.get("identity_profile")
    if isinstance(identity, dict):
        identity.pop("age", None)


def prune_deprecated_registry_fields(registry: Dict[str, Any]) -> None:
    for character in registry.get("characters", {}).values():
        if isinstance(character, dict):
            prune_deprecated_character_fields(character)


def _refresh_record_voice_profiles(book_slug: str, record: Dict[str, Any]) -> None:
    role_id = str(record.get("role_id", "character"))
    display_name = str(record.get("display_name", role_id))
    seed = int(record.get("voice_identity", {}).get("seed", role_seed(book_slug, role_id)))
    differentiators = list(
        record.get("voice_identity", {}).get("differentiators", choose_differentiators(book_slug, role_id))
    )
    voice = build_compact_voice_profile(display_name, {"identity_profile": record.get("identity_profile", {})})
    voice["qwen_instruct"] = append_differentiators(str(voice["qwen_instruct"]), differentiators)
    variants = record.setdefault("voice_variants", {})
    variants["default"] = {
        **dict(variants.get("default", {})),
        "role_id": f"{role_id}_default",
        "display_name": f"{display_name}_default",
        "voice_identity": {"seed": seed, "differentiators": differentiators},
        "voice_profile": voice,
        "voice_config_path": variants.get("default", {}).get("voice_config_path"),
        "voice_config_hash": None,
    }
    variants["internal"] = {
        **dict(variants.get("internal", {})),
        "role_id": f"{role_id}_internal",
        "display_name": f"{display_name}_internal",
        "voice_identity": {"seed": seed, "differentiators": differentiators},
        "voice_profile": _internal_voice_profile(display_name, voice),
        "voice_config_path": variants.get("internal", {}).get("voice_config_path"),
        "voice_config_hash": None,
    }


def build_compact_voice_profile(display_name: str, profile: Dict[str, Any]) -> Dict[str, str]:
    identity = dict(profile.get("identity_profile", profile))
    age_stage = str(identity.get("age_stage", "unknown")).replace("_", " ")
    gender = str(identity.get("gender", "unknown"))
    personality = _string_list(identity.get("personality"))
    accent = _nullable_string(identity.get("accent"))
    race_or_ethnicity = _nullable_string(identity.get("race_or_ethnicity"))

    age_gender_parts: List[str] = []
    if age_stage != "unknown":
        age_gender_parts.append(age_stage)
    if gender != "unknown":
        age_gender_parts.append(gender)
    identity_phrase = " ".join(age_gender_parts).strip() or f"{display_name} voice"
    personality_phrase = ", ".join(personality) if personality else "natural"

    description_parts = [identity_phrase, personality_phrase]
    if race_or_ethnicity:
        description_parts.append(str(race_or_ethnicity))
    if accent:
        description_parts.append(f"{accent} accent")

    qwen_parts = [f"A {identity_phrase} voice", f"{personality_phrase} personality"]
    if accent:
        qwen_parts.append(f"{accent} accent")
    qwen_parts.append("clear natural audiobook delivery")
    return {
        "description": "; ".join(description_parts),
        "qwen_instruct": "; ".join(qwen_parts) + ".",
    }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _nullable_string(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe_preserving_order(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        normalized = normalize_name(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped
