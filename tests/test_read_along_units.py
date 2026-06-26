from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.read_along.units import build_read_along_units


def test_book_paths_exposes_read_along_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")

    assert paths.read_along_units("chapter_015") == (
        tmp_path / "book" / "read_along" / "chapter_015.units.json"
    )
    assert paths.read_along_session_dir("session-1") == (
        tmp_path / "book" / "read_along_sessions" / "session-1"
    )
    assert paths.read_along_timing_log("session-1") == (
        tmp_path / "book" / "read_along_sessions" / "session-1" / "timings.jsonl"
    )


def test_build_read_along_units_preserves_quote_offsets():
    text = 'Leigh said, "Right." Then she left.'
    extraction = extract_quoted_dialogue(text)
    attribution = QuoteAttributionResult(
        roles=["leigh_adult"],
        quotes=[(1, 0, "dialogue")],
    )

    units = build_read_along_units(
        chapter="chapter_015",
        chapter_text=text,
        extraction=extraction,
        attribution=attribution,
        registry=_registry_with_voices(),
        temp_registry={},
    )

    assert [unit.text for unit in units] == ["Leigh said,", '"Right."', "Then she left."]
    assert units[0].role_id == "narrator"
    assert units[1].role_id == "leigh_adult"
    assert text[units[1].source_start:units[1].source_end] == '"Right."'
    assert units[1].voice_config_path == "voices/leigh_adult.qvp"


def test_narrator_quote_uses_book_narrator_and_ignores_role_index():
    text = 'The sign said "Closed" on the door.'
    extraction = extract_quoted_dialogue(text)
    attribution = QuoteAttributionResult(
        roles=["leigh_adult"],
        quotes=[(1, 0, "narrator_quote")],
    )

    units = build_read_along_units(
        chapter="chapter_015",
        chapter_text=text,
        extraction=extraction,
        attribution=attribution,
        registry=_registry_with_voices(),
        temp_registry={},
    )

    closed = [unit for unit in units if unit.text == '"Closed"'][0]
    assert closed.role == "Narrator"
    assert closed.role_id == "narrator"
    assert closed.type == "narration"
    assert closed.voice_variant is None
    assert closed.voice_config_path == "voices/narrator.qvp"


def _registry_with_voices():
    return {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
            "voice_profile": {"qwen_instruct": "male narrator", "description": "male narrator"},
            "voice_identity": {"seed": 1, "differentiators": []},
        },
        "characters": {
            "leigh_adult": {
                "role_id": "leigh_adult",
                "profile_id": "leigh_adult",
                "person_id": "leigh",
                "display_name": "Leigh",
                "age_stage": "adult",
                "aliases": [],
                "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
                "voice_identity": {"seed": 2, "differentiators": []},
                "voice_profile": {"qwen_instruct": "adult female", "description": "adult female"},
                "voice_config_path": "voices/leigh_adult.qvp",
            }
        },
    }
