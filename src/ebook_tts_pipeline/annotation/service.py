from __future__ import annotations

from typing import Dict, List, Set

from ebook_tts_pipeline.annotation.anthropic_client import JsonCompletionClient
from ebook_tts_pipeline.annotation.prompts import (
    SYSTEM_PROMPT,
    render_annotation_prompt,
    render_repair_prompt,
)
from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.domain import AnnotationResult, Sentence


class AnnotationService:
    def __init__(self, client: JsonCompletionClient, repair_retries: int) -> None:
        self.client = client
        self.repair_retries = repair_retries

    def annotate_window(
        self,
        chapter: str,
        sentences: List[Sentence],
        registry: Dict,
    ) -> AnnotationResult:
        prompt = render_annotation_prompt(chapter, sentences, registry)
        payload = self.client.complete_json(SYSTEM_PROMPT, prompt)
        result = AnnotationResult.from_dict(payload)
        expected = [sentence.idx for sentence in sentences]
        known_names = _known_names(registry)

        for attempt in range(self.repair_retries + 1):
            try:
                validate_annotation(
                    result,
                    expected_sentence_indices=expected,
                    known_names=known_names,
                )
                return result
            except AnnotationValidationError as exc:
                if attempt >= self.repair_retries:
                    raise
                repair_prompt = render_repair_prompt(prompt, result.to_dict(), str(exc))
                payload = self.client.complete_json(SYSTEM_PROMPT, repair_prompt)
                result = AnnotationResult.from_dict(payload)

        return result


def _known_names(registry: Dict) -> Set[str]:
    names = {"Narrator"}
    for character in registry.get("characters", {}).values():
        display_name = str(character.get("display_name", ""))
        if display_name:
            names.add(display_name)
        names.update(str(alias) for alias in character.get("aliases", []))
    return names
