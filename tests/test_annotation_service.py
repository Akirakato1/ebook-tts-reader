import json

from ebook_tts_pipeline.annotation.service import AnnotationService
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
