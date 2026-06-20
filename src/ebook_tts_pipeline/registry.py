from __future__ import annotations

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
            registry["characters"][role_id] = {
                "role_id": role_id,
                "display_name": name,
                "aliases": [],
                "character_profile": character.get("profile", {}),
                "voice_identity": {
                    "seed": role_seed(book_slug, role_id),
                    "differentiators": differentiators,
                },
                "voice_profile": voice,
                "voice_config_path": None,
                "first_seen": chapter,
            }
            normalized_known.add(normalized)

        self.save(registry)
