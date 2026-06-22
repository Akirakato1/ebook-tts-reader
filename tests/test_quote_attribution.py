import pytest

from ebook_tts_pipeline.annotation.quote_attribution import (
    QuoteAttributionResult,
    QuoteAttributionService,
    QuoteAttributionValidationError,
    render_quote_attribution_prompt,
    validate_quote_attribution,
)
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue


class FakeQuoteClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.payload


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
            "quotes": [[1, 0, "dialogue"], [2, 1, "dialogue"]],
        }
    )

    validate_quote_attribution(result, quote_indices=[1, 2], known_role_ids={"callie_child"})


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
