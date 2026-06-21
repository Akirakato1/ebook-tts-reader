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
    known_characters = registry.get("characters", {})
    return (
        f"Book title: {book_title}\n\n"
        f"Existing registry: {json.dumps(known_characters, ensure_ascii=False)}\n\n"
        "Build a canonical character registry for audiobook voice casting.\n"
        "Do not produce sentence-level annotation or script rows.\n"
        "Merge aliases that clearly refer to the same person, such as first name, full name, title, or nickname.\n"
        "Create separate profiles only when the same person appears at a different life stage: child, teen, adult, or elder.\n"
        "Return JSON with exactly this shape: {\"characters\":[{\"name\":str,\"profile\":object,\"evidence\":list}]}.\n"
        "Each profile must include age_stage, gender, personality.\n"
        "Profile optional fields: profile_id, person_id, age, race_or_ethnicity, accent, timeline, aliases, same_person_as, narrative_notes.\n"
        "Keep personality to short trait adjectives useful for voice casting.\n"
        "Use race_or_ethnicity and accent only when explicit or strongly text-grounded; otherwise null or omit.\n"
        "Evidence should be compact chapter references and short identity notes.\n\n"
        f"Chapter text:\n{rendered_chapters}\n\n"
        "Return JSON only. Do not wrap the JSON in Markdown code fences."
    )
