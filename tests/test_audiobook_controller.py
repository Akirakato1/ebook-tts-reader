from pathlib import Path

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.read_along.narrator_profile import narrator_profile_hash
from ebook_tts_pipeline.registry import RegistryManager
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.ui.controller import PrototypeUiController


class BatchRecordingFakeTtsAdapter(FakeTtsAdapter):
    def __init__(self) -> None:
        super().__init__(sample_rate=1000, samples_per_character=1)
        self.batch_calls = []
        self.batch_jobs = []
        self.close_calls = 0

    def generate_sentence_batches(self, jobs):
        self.batch_calls.append([int(job["unit_idx"]) for job in jobs])
        self.batch_jobs.append([dict(job) for job in jobs])
        yield self.generate_sentences(jobs)

    def close(self) -> None:
        self.close_calls += 1


class AudiobookControllerPipeline:
    def __init__(self, config, adapter) -> None:
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.tts_adapter = adapter

    def build_read_along_units(self, chapter):
        return read_json(self.paths.read_along_units(chapter))["units"]


def test_controller_generates_persistent_audiobook_chapter_with_large_windows(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_narrator_only_book(paths, unit_count=5)
    adapter = BatchRecordingFakeTtsAdapter()

    def factory(config, needs_llm, fake_tts):
        return AudiobookControllerPipeline(config, adapter)

    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=factory, fake_tts=True)

    result = controller.generate_audiobook_chapters(
        ["chapter_001"],
        {
            "generation_mode": "balanced",
            "model_profile": "12hz",
            "max_window_chars": 6000,
        },
        force=True,
    )

    assert result["chapters"] == 1
    assert paths.audiobook_chapter_audio("chapter_001").exists()
    assert paths.audiobook_chapter_timeline("chapter_001").exists()
    assert adapter.batch_calls == [[0, 1, 2, 3, 4]]
    assert adapter.close_calls == 1
    manifest = read_json(paths.audiobook_manifest)
    chapter = manifest["chapters"]["chapter_001"]
    assert chapter["audio_path"] == "audiobook/chapter_001.wav"
    assert chapter["window_count"] == 1
    assert chapter["unit_count"] == 5
    assert chapter["settings"]["model_profile"] == "12hz"


def test_controller_skips_existing_audiobook_without_clobbering_manifest(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_narrator_only_book(paths, unit_count=5)
    adapter = BatchRecordingFakeTtsAdapter()

    def factory(config, needs_llm, fake_tts):
        return AudiobookControllerPipeline(config, adapter)

    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=factory, fake_tts=True)
    controller.generate_audiobook_chapters(["chapter_001"], {"generation_mode": "balanced"}, force=True)
    adapter.batch_calls.clear()

    result = controller.generate_audiobook_chapters(["chapter_001"], {"generation_mode": "balanced"}, force=False)

    assert result["skipped"] == 1
    assert adapter.batch_calls == []
    chapter = read_json(paths.audiobook_manifest)["chapters"]["chapter_001"]
    assert chapter["window_count"] == 1
    assert chapter["unit_count"] == 5


def test_controller_audiobook_generation_uses_separate_audiobook_narrator_profile(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_narrator_only_book(paths, unit_count=2)
    adapter = BatchRecordingFakeTtsAdapter()

    def factory(config, needs_llm, fake_tts):
        return AudiobookControllerPipeline(config, adapter)

    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=factory, fake_tts=True)
    controller.save_read_along_narrator_profile(
        {
            "display_name": "Read Along Narrator",
            "age_stage": "adult",
            "gender": "male",
            "personality": "steady",
            "race_or_ethnicity": "White",
            "accent": "American",
            "occupation": "read-along narrator",
        }
    )
    audiobook_profile = controller.save_audiobook_narrator_profile(
        {
            "display_name": "Audiobook Narrator",
            "age_stage": "adult",
            "gender": "female",
            "personality": "warm, crisp",
            "race_or_ethnicity": "White",
            "accent": "American",
            "occupation": "audiobook narrator",
        }
    )

    controller.generate_audiobook_chapters(["chapter_001"], {"generation_mode": "balanced"}, force=True)

    expected_hash = narrator_profile_hash(audiobook_profile)
    used_voice_paths = {str(job["voice_config_path"]) for job in adapter.batch_jobs[0]}
    assert used_voice_paths == {f"voices/_narrator/{expected_hash}/narrator.qvp"}
    assert read_json(paths.read_along_narrator_profile)["display_name"] == "Read Along Narrator"
    assert read_json(paths.audiobook_narrator_profile)["display_name"] == "Audiobook Narrator"


def _write_narrator_only_book(paths: BookPaths, unit_count: int) -> None:
    paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").write_text(" ".join(f"Sentence {index}." for index in range(unit_count)), encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "book"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_identity": {"seed": 1, "differentiators": []},
            },
            "characters": {},
        },
    )
    units = []
    for index in range(unit_count):
        text = f"Sentence {index}."
        units.append(
            {
                "chapter": "chapter_001",
                "unit_id": index,
                "text": text,
                "source_start": index * 12,
                "source_end": index * 12 + len(text),
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "voice_config_path": "voices/narrator.qvp",
                "quote_id": None,
                "sentence_idx": index,
                "character": None,
                "voice_variant": None,
            }
        )
    write_json_atomic(paths.read_along_units("chapter_001"), {"chapter": "chapter_001", "units": units})
