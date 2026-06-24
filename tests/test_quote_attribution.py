import pytest

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.quote_attribution import (
    QuoteAttributionResult,
    QuoteAttributionService,
    QuoteAttributionValidationError,
    render_quote_attribution_prompt,
    validate_quote_attribution,
)
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction, QuoteSpan, extract_quoted_dialogue
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.json_io import read_json


class FakeQuoteClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.payload


class SequenceQuoteClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return payload


class BadQuoteClient:
    def complete_json(self, system_prompt, user_prompt):
        return {"roles": ["Narrator"], "quotes": []}


def test_quote_attribution_prompt_uses_marked_quotes_and_registry_role_ids():
    extraction = extract_quoted_dialogue('Callie said, "Stay here."')
    registry = {
        "characters": {
            "callie_child": {
                "role_id": "callie_child",
                "display_name": "Callie",
                "age_stage": "child",
                "identity_profile": {
                    "age_stage": "child",
                    "gender": "female",
                    "race_or_ethnicity": None,
                    "accent": None,
                    "occupation": None,
                    "personality": ["guarded"],
                },
                "aliases": ["Callie child"],
            }
        }
    }

    prompt = render_quote_attribution_prompt(
        chapter="chapter_001",
        extraction=extraction,
        registry=registry,
    )

    assert '|q001| "Stay here." ||q001||' in prompt
    assert '"role_id": "callie_child"' in prompt
    assert "local_speakers" in prompt
    assert "Do not create global registry characters" in prompt
    assert "Do not label normal quoted dialogue as Narrator" in prompt
    assert "Omit the third item for normal dialogue" in prompt
    assert "Do not reuse one local speaker for distinct unnamed people" in prompt
    assert "Every local_speakers entry must be assigned to at least one quote" in prompt
    assert "narrator_quote is a quote type only" in prompt
    assert "Use the exact local_id in roles" in prompt
    assert "security_guard_001" in prompt
    assert "Step 1: Review the global registry" in prompt
    assert "Step 2: Build the chapter-local registry" in prompt
    assert "Step 3: Build roles from the union" in prompt
    assert "Step 4: Assign every quote" in prompt
    assert "local_speakers is the local registry for this chapter only" in prompt
    assert "roles must contain only global role_id values and local_speakers.local_id values" in prompt
    assert "Use local_speakers only for real character dialogue by an agentic speaker" in prompt
    assert "automated phone-system messages" in prompt
    assert "Mark those rows as narrator_quote instead" in prompt


def test_quote_attribution_validator_accepts_registry_and_local_speakers():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child", "local_001"],
            "local_speakers": [
                {
                    "local_id": "local_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "race_or_ethnicity": None,
                        "accent": None,
                        "occupation": "security guard",
                        "personality": ["brusque"],
                    },
                }
            ],
            "quotes": [[1, 0], [2, 1]],
        }
    )

    validate_quote_attribution(result, quote_indices=[1, 2], known_role_ids={"callie_child"})


def test_quote_attribution_result_serializes_dialogue_rows_compactly():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child", "Narrator"],
            "quotes": [[1, 0, "dialogue"], [2, 1, "narrator_quote"]],
        }
    )

    assert result.to_dict()["quotes"] == [[1, 0], [2, 1, "narrator_quote"]]


def test_quote_attribution_result_accepts_model_rows_with_quote_and_role_ids():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child", "local_001"],
            "local_speakers": [
                {
                    "local_id": "local_001",
                    "label": "Security Guard",
                    "profile": {"age_stage": "adult", "gender": "unknown", "personality": ["brusque"]},
                }
            ],
            "quotes": [["q001", "callie_child", "dialogue"], ["q002", "local_001", "dialogue"]],
        }
    )

    assert result.quotes == [(1, 0, "dialogue"), (2, 1, "dialogue")]


def test_quote_attribution_result_accepts_model_rows_with_role_id_first():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["local_001"],
            "local_speakers": [
                {
                    "local_id": "local_001",
                    "label": "Security Guard",
                    "profile": {"age_stage": "adult", "gender": "unknown", "personality": ["brusque"]},
                }
            ],
            "quotes": [["local_001", "q001", "dialogue"]],
        }
    )

    assert result.quotes == [(1, 0, "dialogue")]


def test_quote_attribution_result_accepts_model_rows_with_extra_quote_text():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child"],
            "quotes": [
                [1, 0, "Stay here.", "dialogue"],
                [2, 0, "A quoted term without explicit type"],
            ],
        }
    )

    assert result.quotes == [(1, 0, "dialogue"), (2, 0, "dialogue")]


def test_quote_attribution_validator_rejects_missing_quote_assignment():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child"],
            "local_speakers": [],
            "quotes": [[1, 0, "dialogue"]],
        }
    )

    with pytest.raises(QuoteAttributionValidationError, match="missing quote assignments"):
        validate_quote_attribution(result, quote_indices=[1, 2], known_role_ids={"callie_child"})


def test_quote_attribution_validator_rejects_unused_local_speakers():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child", "local_001"],
            "local_speakers": [
                {
                    "local_id": "local_001",
                    "label": "Security Guard",
                    "profile": {"age_stage": "adult"},
                }
            ],
            "quotes": [[1, 0, "dialogue"]],
        }
    )

    with pytest.raises(QuoteAttributionValidationError, match="unused local speakers: local_001"):
        validate_quote_attribution(result, quote_indices=[1], known_role_ids={"callie_child"})


def test_quote_attribution_validator_rejects_narrator_dialogue():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["Narrator"],
            "local_speakers": [],
            "quotes": [[1, 0, "dialogue"]],
        }
    )

    with pytest.raises(QuoteAttributionValidationError, match="Narrator cannot speak dialogue"):
        validate_quote_attribution(result, quote_indices=[1], known_role_ids=set())


def test_quote_attribution_validator_ignores_role_for_narrator_quote():
    result = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child"],
            "local_speakers": [],
            "quotes": [[1, 0, "narrator_quote"]],
        }
    )

    validate_quote_attribution(result, quote_indices=[1], known_role_ids={"callie_child"})


def test_quote_attribution_service_calls_client_and_validates_result():
    extraction = extract_quoted_dialogue('"Move along."')
    client = FakeQuoteClient(
        {
            "roles": ["local_001"],
            "local_speakers": [
                {
                    "local_id": "local_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["brusque"],
                    },
                }
            ],
            "quotes": [[1, 0, "dialogue"]],
        }
    )
    service = QuoteAttributionService(client)

    result = service.attribute_quotes(
        chapter="chapter_001",
        extraction=extraction,
        registry={"characters": {}},
    )

    assert result.roles == ["local_001"]
    assert result.quotes == [(1, 0, "dialogue")]
    assert client.calls


def test_quote_attribution_service_auto_repairs_narrator_quote_role_confusion_without_llm_retry():
    extraction = extract_quoted_dialogue('They called it "evidence" and then "closure."')
    client = SequenceQuoteClient(
        [
            {
                "roles": ["narrator_quote"],
                "quotes": [[1, 0, "narrator_quote"], [2, 0, "narrator_quote"]],
            }
        ]
    )
    service = QuoteAttributionService(client)

    result = service.attribute_quotes(
        chapter="chapter_002",
        extraction=extraction,
        registry={"characters": {}},
    )

    assert result.roles == ["Narrator"]
    assert result.quotes == [(1, 0, "narrator_quote"), (2, 0, "narrator_quote")]
    assert len(client.calls) == 1


def test_quote_attribution_service_canonicalizes_narrator_quote_rows_to_narrator_role():
    extraction = extract_quoted_dialogue('"Dialogue." Then the recording said, "Please hang up."')
    client = SequenceQuoteClient(
        [
            {
                "roles": ["leigh_collier_teen", "callie_dewinter_teen"],
                "quotes": [[1, 0], [2, 0, "narrator_quote"]],
            }
        ]
    )
    service = QuoteAttributionService(client, repair_retries=0)

    result = service.attribute_quotes(
        chapter="chapter_005",
        extraction=extraction,
        registry={
            "characters": {
                "leigh_collier_teen": {"role_id": "leigh_collier_teen"},
                "callie_dewinter_teen": {"role_id": "callie_dewinter_teen"},
            }
        },
    )

    assert result.roles == ["leigh_collier_teen", "Narrator"]
    assert result.quotes == [(1, 0, "dialogue"), (2, 1, "narrator_quote")]
    assert len(client.calls) == 1


def test_quote_attribution_service_prunes_local_speaker_used_only_by_narrator_quote():
    extraction = extract_quoted_dialogue('"If you would like to make a call ..." "Please hang up."')
    client = SequenceQuoteClient(
        [
            {
                "roles": ["phone_recording_001"],
                "local_speakers": [
                    {
                        "local_id": "phone_recording_001",
                        "label": "Phone Recording",
                        "profile": {
                            "age_stage": "unknown",
                            "gender": "unknown",
                            "personality": ["flat", "automated"],
                        },
                    }
                ],
                "quotes": [[1, 0, "narrator_quote"], [2, 0, "narrator_quote"]],
            }
        ]
    )
    service = QuoteAttributionService(client, repair_retries=0)

    result = service.attribute_quotes(
        chapter="chapter_005",
        extraction=extraction,
        registry={"characters": {}},
    )

    assert result.roles == ["Narrator"]
    assert result.quotes == [(1, 0, "narrator_quote"), (2, 0, "narrator_quote")]
    assert result.local_speakers == []
    assert len(client.calls) == 1


def test_quote_attribution_service_rejects_local_speaker_label_used_as_role_id():
    extraction = extract_quoted_dialogue('The guard said, "Move along."')
    client = SequenceQuoteClient(
        [
            {
                "roles": ["Security Guard"],
                "local_speakers": [
                    {
                        "local_id": "local_001",
                        "label": "Security Guard",
                        "profile": {
                            "age_stage": "adult",
                            "gender": "male",
                            "occupation": "security guard",
                            "personality": ["brusque"],
                        },
                    }
                ],
                "quotes": [[1, 0, "dialogue"]],
            }
        ]
    )
    service = QuoteAttributionService(client, repair_retries=0)

    with pytest.raises(QuoteAttributionValidationError, match="local role missing profile"):
        service.attribute_quotes(
            chapter="chapter_002",
            extraction=extraction,
            registry={"characters": {}},
        )
    assert len(client.calls) == 1


def test_quote_attribution_service_repairs_missing_assignment_with_llm_once():
    extraction = extract_quoted_dialogue('"One." "Two."')
    client = SequenceQuoteClient(
        [
            {
                "roles": ["Narrator"],
                "quotes": [[1, 0, "narrator_quote"]],
            },
            {
                "roles": ["Narrator"],
                "quotes": [[1, 0, "narrator_quote"], [2, 0, "narrator_quote"]],
            },
        ]
    )
    service = QuoteAttributionService(client)

    result = service.attribute_quotes(
        chapter="chapter_002",
        extraction=extraction,
        registry={"characters": {}},
    )

    assert result.quotes == [(1, 0, "narrator_quote"), (2, 0, "narrator_quote")]
    assert len(client.calls) == 2
    assert "missing quote assignments: [2]" in client.calls[1]["user"]


def test_quote_attribution_service_repairs_non_json_model_response_with_raw_text():
    extraction = extract_quoted_dialogue('"Stay here."')
    client = SequenceQuoteClient(
        [
            AnnotationModelOutputError(
                "Anthropic response was not valid JSON",
                raw_text="I'll work through this systematically before returning the JSON.",
                source="Anthropic response",
            ),
            {
                "roles": ["callie_child"],
                "quotes": [[1, 0, "dialogue"]],
            },
        ]
    )
    service = QuoteAttributionService(client, repair_retries=1)

    result = service.attribute_quotes(
        chapter="chapter_005",
        extraction=extraction,
        registry={"characters": {"callie_child": {"role_id": "callie_child"}}},
    )

    assert result.quotes == [(1, 0, "dialogue")]
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["user"]
    assert "previous response was not valid JSON" in repair_prompt
    assert "I'll work through this systematically" in repair_prompt
    assert "Return corrected JSON only" in repair_prompt


def test_quote_attribution_service_logs_validation_failure(tmp_path):
    extraction = QuoteExtraction(
        text='"Hello."',
        quotes=[QuoteSpan(idx=1, quote_id="q001", start=0, end=8, text='"Hello."')],
        narrator_spans=[],
    )
    logger = FailureLogger(tmp_path / "failures", context={"book_root": "book"})
    service = QuoteAttributionService(BadQuoteClient(), failure_logger=logger, repair_retries=0)

    with pytest.raises(QuoteAttributionValidationError, match="missing quote assignments"):
        service.attribute_quotes("chapter_002", extraction, {"characters": {}})

    logs = list((tmp_path / "failures").glob("*.json"))
    assert len(logs) == 1
    payload = read_json(logs[0])
    assert payload["event_type"] == "quote_attribution_validation_failed"
    assert payload["context"]["chapter"] == "chapter_002"
    assert payload["details"]["quote_ids"] == ["q001"]
    assert payload["details"]["attempt"] == 0
    assert payload["details"]["repair_available"] is False
    assert "Chapter: chapter_002" in payload["details"]["user_prompt"]


def test_quote_attribution_service_logs_repair_exhaustion(tmp_path):
    extraction = QuoteExtraction(
        text='"Hello."',
        quotes=[QuoteSpan(idx=1, quote_id="q001", start=0, end=8, text='"Hello."')],
        narrator_spans=[],
    )
    logger = FailureLogger(tmp_path / "failures", context={"book_root": "book"})
    client = SequenceQuoteClient(
        [
            {"roles": ["Narrator"], "quotes": []},
            {"roles": ["Narrator"], "quotes": []},
        ]
    )
    service = QuoteAttributionService(client, failure_logger=logger, repair_retries=1)

    with pytest.raises(QuoteAttributionValidationError, match="missing quote assignments"):
        service.attribute_quotes("chapter_002", extraction, {"characters": {}})

    logs = sorted((tmp_path / "failures").glob("*.json"))
    assert len(logs) == 2
    first = read_json(logs[0])
    second = read_json(logs[1])
    assert first["details"]["attempt"] == 0
    assert first["details"]["repair_available"] is True
    assert second["details"]["attempt"] == 1
    assert second["details"]["repair_available"] is False
    assert "The previous JSON failed validation" in second["details"]["user_prompt"]
    assert len(client.calls) == 2
