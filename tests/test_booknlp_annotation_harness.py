from ebook_tts_pipeline.annotation.booknlp_runner import BookNlpRunner, BookNlpRunnerConfig
from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_consolidation import BookNlpSonnetConsolidationService
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths


def test_booknlp_runner_reuses_cache_when_input_hash_matches(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").write_text('One. "Hello."', encoding="utf-8")
    calls = []

    def fake_run(command):
        calls.append(command)

    runner = BookNlpRunner(BookNlpRunnerConfig(python="python", model="small"), run_command=fake_run)

    first = runner.ensure_booknlp_artifacts(paths, ["chapter_001"])
    second = runner.ensure_booknlp_artifacts(paths, ["chapter_001"])

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1
    assert paths.booknlp_input.read_text(encoding="utf-8") == "[chapter_001]\nOne. \"Hello.\""
    manifest = read_json(paths.booknlp_manifest)
    assert manifest["model"] == "small"
    assert manifest["chapter_count"] == 1


class NoCallClient:
    def complete_json(self, system_prompt, user_prompt):
        raise AssertionError("Sonnet should not be called for deterministic mappings")


def test_harness_service_writes_valid_annotation_without_sonnet_for_unique_match(tmp_path):
    chapter_text = 'Mary paused. "The apple of my eye," Mr. Pounds said.'
    extraction = extract_quoted_dialogue(chapter_text)
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye,"',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]
    service = BookNlpSonnetConsolidationService(NoCallClient())

    result = service.consolidate("chapter_017", extraction, candidates, registry)

    assert result.to_dict() == {
        "roles": ["mr_john_pounds_adult"],
        "quotes": [[1, 0]],
    }
