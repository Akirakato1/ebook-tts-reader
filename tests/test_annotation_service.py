import json

import pytest

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import Sentence


class FakeLlmClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        payload = self.payloads.pop(0)
        if isinstance(payload, str):
            return json.loads(payload)
        return payload


class FailingLlmClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        raise AnnotationModelOutputError(
            "Anthropic response was not valid JSON",
            raw_text="not json from model",
            source="Anthropic response",
        )


def test_annotation_service_returns_valid_result_without_repair():
    client = FakeLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [0, 0, 1]],
            }
        ]
    )
    service = AnnotationService(client=client, repair_retries=1)

    result = service.annotate_window(
        chapter="chapter_001",
        sentences=[Sentence(0, "It rained."), Sentence(1, "The road shone.")],
        registry={"characters": {}},
    )

    assert result.roles == ["Narrator"]
    assert result.script == [(0, 0, 0), (0, 0, 1)]
    assert len(client.calls) == 1


def test_annotation_service_repairs_invalid_result_once():
    client = FakeLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0]],
            },
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [0, 0, 1]],
            },
        ]
    )
    service = AnnotationService(client=client, repair_retries=1)

    result = service.annotate_window(
        chapter="chapter_001",
        sentences=[Sentence(0, "It rained."), Sentence(1, "The road shone.")],
        registry={"characters": {}},
    )

    assert result.script == [(0, 0, 0), (0, 0, 1)]
    assert "missing sentence indexes: [1]" in client.calls[1]["user"]


def test_annotation_service_logs_model_output_failure(tmp_path):
    logger = FailureLogger(tmp_path / "logs", context={"book_root": "books/demo"})
    service = AnnotationService(
        client=FailingLlmClient(),
        repair_retries=0,
        failure_logger=logger,
    )

    with pytest.raises(AnnotationModelOutputError) as exc:
        service.annotate_window(
            chapter="chapter_001",
            sentences=[Sentence(0, "It rained.")],
            registry={"characters": {}},
        )

    logs = list((tmp_path / "logs").glob("*.json"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert "annotation_model_output_error" in log_text
    assert "chapter_001" in log_text
    assert "It rained." in log_text
    assert "not json from model" in log_text
    assert exc.value.debug_log_path == str(logs[0])


def test_annotation_service_logs_validation_failure_before_repair(tmp_path):
    client = FakeLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0]],
            },
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [0, 0, 1]],
            },
        ]
    )
    service = AnnotationService(
        client=client,
        repair_retries=1,
        failure_logger=FailureLogger(tmp_path / "logs"),
    )

    result = service.annotate_window(
        chapter="chapter_001",
        sentences=[Sentence(0, "It rained."), Sentence(1, "The road shone.")],
        registry={"characters": {}},
    )

    logs = list((tmp_path / "logs").glob("*.json"))
    assert result.script == [(0, 0, 0), (0, 0, 1)]
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert "annotation_validation_failed" in log_text
    assert "missing sentence indexes: [1]" in log_text
    assert "The road shone." in log_text
