from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from ebook_tts_pipeline.annotation.anthropic_client import (
    AnnotationModelOutputError,
    JsonCompletionClient,
)
from ebook_tts_pipeline.annotation.prompts import (
    SYSTEM_PROMPT,
    render_annotation_prompt,
    render_repair_prompt,
)
from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult, Sentence


class AnnotationService:
    def __init__(
        self,
        client: JsonCompletionClient,
        repair_retries: int,
        failure_logger: Optional[FailureLogger] = None,
    ) -> None:
        self.client = client
        self.repair_retries = repair_retries
        self.failure_logger = failure_logger

    def annotate_window(
        self,
        chapter: str,
        sentences: List[Sentence],
        registry: Dict,
        lock_registry: bool = False,
    ) -> AnnotationResult:
        prompt = render_annotation_prompt(chapter, sentences, registry, lock_registry=lock_registry)
        payload = self._complete_json(chapter, sentences, "annotation", prompt)
        result = self._annotation_result_from_payload(chapter, sentences, "annotation", prompt, payload)
        result = _lock_annotation_result(result) if lock_registry else result
        expected = [sentence.idx for sentence in sentences]
        known_names = known_annotation_role_names(registry)

        for attempt in range(self.repair_retries + 1):
            try:
                validate_annotation(
                    result,
                    expected_sentence_indices=expected,
                    known_names=known_names,
                )
                return result
            except AnnotationValidationError as exc:
                self._log_failure(
                    "annotation_validation_failed",
                    chapter=chapter,
                    sentences=sentences,
                    prompt=prompt,
                    exc=exc,
                    details={
                        "attempt": attempt,
                        "repair_available": attempt < self.repair_retries,
                        "payload": result.to_dict(),
                    },
                )
                if attempt >= self.repair_retries:
                    raise
                repair_prompt = render_repair_prompt(prompt, result.to_dict(), str(exc))
                payload = self._complete_json(chapter, sentences, "repair", repair_prompt)
                result = self._annotation_result_from_payload(
                    chapter,
                    sentences,
                    "repair",
                    repair_prompt,
                    payload,
                )
                result = _lock_annotation_result(result) if lock_registry else result

        return result

    def _complete_json(
        self,
        chapter: str,
        sentences: List[Sentence],
        call_type: str,
        prompt: str,
    ) -> Dict:
        try:
            return self.client.complete_json(SYSTEM_PROMPT, prompt)
        except Exception as exc:
            self._log_failure(
                "annotation_model_output_error",
                chapter=chapter,
                sentences=sentences,
                prompt=prompt,
                exc=exc,
                details={
                    "call_type": call_type,
                    "source": getattr(exc, "source", None),
                    "raw_model_text": getattr(exc, "raw_text", None),
                },
            )
            raise

    def _annotation_result_from_payload(
        self,
        chapter: str,
        sentences: List[Sentence],
        call_type: str,
        prompt: str,
        payload: Dict,
    ) -> AnnotationResult:
        try:
            return AnnotationResult.from_dict(payload)
        except Exception as exc:
            wrapped = AnnotationModelOutputError(f"Annotation JSON did not match schema: {exc}")
            self._log_failure(
                "annotation_payload_invalid",
                chapter=chapter,
                sentences=sentences,
                prompt=prompt,
                exc=wrapped,
                details={
                    "call_type": call_type,
                    "payload": payload,
                },
            )
            raise wrapped from exc

    def _log_failure(
        self,
        event_type: str,
        chapter: str,
        sentences: List[Sentence],
        prompt: str,
        exc: BaseException,
        details: Dict[str, Any],
    ) -> None:
        if self.failure_logger is None:
            return
        indexes = [sentence.idx for sentence in sentences]
        log_details: Dict[str, Any] = {
            "chapter": chapter,
            "sentence_count": len(sentences),
            "sentence_indices": indexes,
            "first_sentence_idx": indexes[0] if indexes else None,
            "last_sentence_idx": indexes[-1] if indexes else None,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt,
        }
        log_details.update(details)
        self.failure_logger.with_context(chapter=chapter).write_failure(
            event_type,
            log_details,
            exc=exc,
        )


def known_annotation_role_names(registry: Dict) -> Set[str]:
    names = {"Narrator"}
    characters = [
        character
        for character in registry.get("characters", {}).values()
        if isinstance(character, dict)
    ]
    display_counts = Counter(
        _normalize_name(str(character.get("display_name", "")))
        for character in characters
        if character.get("display_name")
    )
    for character in characters:
        display_name = str(character.get("display_name", ""))
        if display_name and display_counts[_normalize_name(display_name)] == 1:
            names.add(display_name)
        names.update(str(alias) for alias in character.get("aliases", []))
    return names


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _lock_annotation_result(result: AnnotationResult) -> AnnotationResult:
    if not result.new_characters:
        return result
    return replace(
        result,
        new_characters=[],
        proposed_new_characters=list(result.proposed_new_characters) + list(result.new_characters),
    )
