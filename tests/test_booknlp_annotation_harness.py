from ebook_tts_pipeline.annotation.booknlp_runner import BookNlpRunner, BookNlpRunnerConfig
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
