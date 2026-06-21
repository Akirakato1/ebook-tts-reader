from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.voice_identity import (
    append_differentiators,
    choose_differentiators,
    role_seed,
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
    for character in registry.get("characters", {}).values():
        ensure_character_voice_variants(book_slug, character)
        direct_names = _character_lookup_names(character)
        variant_match = _matching_variant(character, normalized)
        if normalized in direct_names or variant_match:
            variant_key = variant_match or voice_variant_for_type(speech_type)
            variant = character.get("voice_variants", {}).get(variant_key)
            if variant:
                return {
                    "character": str(character.get("display_name", role_name)),
                    "role": str(variant.get("display_name", role_name)),
                    "role_id": str(variant.get("role_id", character.get("role_id", role_name))),
                    "voice_variant": variant_key,
                    "voice_record": variant,
                }
            return {
                "character": str(character.get("display_name", role_name)),
                "role": str(character.get("display_name", role_name)),
                "role_id": str(character.get("role_id", role_name)),
                "voice_variant": None,
                "voice_record": character,
            }

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


def _character_lookup_names(character: Dict[str, Any]) -> Set[str]:
    names = [
        str(character.get("display_name", "")),
        str(character.get("role_id", "")),
        str(character.get("role_id", "")).replace("_", " "),
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
                "character_profile": {"role": "narrator"},
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
        write_json_atomic(self.paths.registry, registry)

    def known_names(self) -> Set[str]:
        registry = self.load()
        names = {"Narrator"}
        for character in registry.get("characters", {}).values():
            display_name = str(character.get("display_name", ""))
            if display_name:
                names.add(display_name)
            names.update(str(alias) for alias in character.get("aliases", []))
        return names

    def add_new_characters(self, chapter: str, new_characters: List[Dict[str, Any]]) -> None:
        registry = self.load()
        normalized_known = {normalize_name(name) for name in self.known_names()}
        book_slug = str(registry["book"]["slug"])

        for character in new_characters:
            name = str(character["name"]).strip()
            normalized = normalize_name(name)
            if normalized in normalized_known:
                raise ValueError(f"collides with existing character or alias: {name}")
            role_id = slugify_name(name)
            differentiators = choose_differentiators(book_slug, role_id)
            voice = dict(character["voice"])
            voice["qwen_instruct"] = append_differentiators(
                str(voice["qwen_instruct"]),
                differentiators,
            )
            seed = role_seed(book_slug, role_id)
            registry["characters"][role_id] = {
                "role_id": role_id,
                "display_name": name,
                "aliases": [],
                "character_profile": character.get("profile", {}),
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
                "first_seen": chapter,
            }
            normalized_known.add(normalized)

        self.save(registry)
