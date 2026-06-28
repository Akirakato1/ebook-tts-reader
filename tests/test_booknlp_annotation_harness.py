from ebook_tts_pipeline.annotation.booknlp_runner import BookNlpRunner, BookNlpRunnerConfig
from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_consolidation import BookNlpSonnetConsolidationService
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths
from scripts.run_booknlp_annotation_harness import build_harness_report, run_cached_harness


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


def test_cached_harness_writes_sidecar_annotation_without_calling_sonnet(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_017").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_017").write_text(
        'Mary paused. "The apple of my eye," Mr. Pounds said.',
        encoding="utf-8",
    )
    paths.booknlp_output_dir.mkdir(parents=True, exist_ok=True)
    (paths.booknlp_output_dir / "book.quotes").write_text(
        "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n"
        "3\t9\t10\t12\tMr. Pounds\t7\tThe apple of my eye,\n",
        encoding="utf-8",
    )
    paths.registry.parent.mkdir(parents=True, exist_ok=True)
    from ebook_tts_pipeline.json_io import write_json_atomic

    write_json_atomic(
        paths.registry,
        {
            "characters": {
                "mr_john_pounds_adult": {
                    "role_id": "mr_john_pounds_adult",
                    "display_name": "Mr John Pounds",
                    "age_stage": "adult",
                    "aliases": ["Mr John Pounds adult"],
                }
            }
        },
    )

    report = run_cached_harness(paths, ["chapter_017"], client=NoCallClient())

    assert report["deterministic_quotes"] == 1
    assert report["sonnet_quotes"] == 0
    assert not paths.annotation("chapter_017").exists()
    sidecar = paths.booknlp_dir / "harness_annotations" / "chapter_017.annotation.json"
    assert read_json(sidecar)["quotes"] == [[1, 0]]


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


def test_harness_report_records_cost_reduction_metrics():
    report = build_harness_report(
        book_slug="victorian_psycho",
        chapters=["chapter_017"],
        deterministic_quotes=8,
        sonnet_quotes=2,
        failed_quotes=0,
        sonnet_prompt_chars=2400,
        old_full_prompt_chars=48000,
    )

    assert report["book_slug"] == "victorian_psycho"
    assert report["chapters"] == ["chapter_017"]
    assert report["deterministic_quotes"] == 8
    assert report["sonnet_quotes"] == 2
    assert report["estimated_prompt_char_savings"] == 45600


def test_booknlp_harness_config_is_opt_in(monkeypatch):
    monkeypatch.delenv("EBOOK_TTS_ANNOTATION_BACKEND", raising=False)
    default_config = PipelineConfig.from_env("book")
    assert default_config.annotation_backend == "sonnet"

    monkeypatch.setenv("EBOOK_TTS_ANNOTATION_BACKEND", "booknlp_harness")
    harness_config = PipelineConfig.from_env("book")
    assert harness_config.annotation_backend == "booknlp_harness"
