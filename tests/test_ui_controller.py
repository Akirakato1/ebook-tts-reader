from pathlib import Path

import pytest

from ebook_tts_pipeline.epub_ingestion import EpubExtractResult
from ebook_tts_pipeline.json_io import write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.ui.controller import ChapterStage, PrototypeUiController


class FakeExtractor:
    def extract(self, epub_path, paths):
        paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
        paths.chapter_text("chapter_002").write_text("Chapter Two\nText.", encoding="utf-8")
        return EpubExtractResult(
            chapters=["chapter_001", "chapter_002"],
            sources=["text/chapter001.xhtml", "text/chapter002.xhtml"],
        )


class FakeRegistry:
    def __init__(self, paths, calls):
        self.paths = paths
        self.calls = calls

    def initialize_if_missing(self, book_title, book_slug):
        self.calls.append(("initialize", book_title, book_slug))
        if not self.paths.registry.exists():
            write_json_atomic(
                self.paths.registry,
                {"book": {"title": book_title, "slug": book_slug}, "characters": {}},
            )


class FakePipeline:
    def __init__(self, config, calls):
        self.paths = BookPaths(config.book_root)
        self.calls = calls
        self.registry = FakeRegistry(self.paths, calls)

    def segment_chapter(self, chapter):
        self.calls.append(("segment", chapter))
        self.paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.sentence_artifact(chapter),
            {"chapter": chapter, "source_path": f"chapters/{chapter}.txt", "sentences": []},
        )

    def annotate_chapter(self, chapter):
        self.calls.append(("annotate", chapter))
        self.paths.annotation(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.annotation(chapter),
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [],
            },
        )

    def build_sentence_jobs(self, chapter, annotation):
        self.calls.append(("build_scripts", chapter))
        self.paths.tts_script(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.paths.tts_script(chapter), {"chapter": chapter, "jobs": []})
        self.paths.qwen_script(chapter).write_text("Narrator: Text.\n", encoding="utf-8")

    def prepare_voices_for_annotation(self, annotation):
        self.calls.append(("prepare_voices",))

    def synthesize_chapter_from_tts_script(self, chapter):
        self.calls.append(("synthesize", chapter))
        self.paths.chapter_audio(chapter).parent.mkdir(parents=True, exist_ok=True)
        self.paths.chapter_audio(chapter).write_bytes(b"wav")


def fake_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts))
        return FakePipeline(config, calls)

    return factory


def test_controller_detects_chapter_stage_from_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\n", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    paths.sentence_artifact("chapter_001").write_text("{}", encoding="utf-8")
    controller = PrototypeUiController(book_root=paths.root)

    assert controller.chapter_rows()[0].stage == ChapterStage.SEGMENTED

    paths.annotation("chapter_001").parent.mkdir(parents=True)
    paths.annotation("chapter_001").write_text("{}", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.ANNOTATED

    paths.tts_script("chapter_001").parent.mkdir(parents=True)
    paths.tts_script("chapter_001").write_text("{}", encoding="utf-8")
    paths.qwen_script("chapter_001").write_text("Narrator: Hi.\n", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.SCRIPTED

    paths.chapter_audio("chapter_001").parent.mkdir(parents=True)
    paths.chapter_audio("chapter_001").write_bytes(b"wav")
    assert controller.chapter_rows()[0].stage == ChapterStage.AUDIO


def test_controller_saves_pretty_registry_json(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root)

    controller.save_registry_text('{"book":{"title":"Demo"},"characters":{}}')

    text = paths.registry.read_text(encoding="utf-8")
    assert '"title": "Demo"' in text

    with pytest.raises(ValueError):
        controller.save_registry_text("{bad json")


def test_controller_loads_epub_initializes_registry_segments_and_toc(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    library_path = tmp_path / "library.json"
    controller = PrototypeUiController(
        book_root=paths.root,
        library_path=library_path,
        pipeline_factory=fake_pipeline_factory(calls),
        extractor=FakeExtractor(),
        fake_tts=True,
    )

    result = controller.load_epub(tmp_path / "demo.epub", title="Demo", slug="demo")

    assert result.chapters == ["chapter_001", "chapter_002"]
    assert paths.registry.exists()
    assert paths.sentence_artifact("chapter_001").exists()
    assert paths.sentence_artifact("chapter_002").exists()
    toc = paths.root / "toc.json"
    assert "Chapter One" in toc.read_text(encoding="utf-8")
    library = controller.library_books()
    assert len(library) == 1
    assert library[0].title == "Demo"
    assert library[0].slug == "demo"
    assert library[0].book_root == paths.root
    assert library[0].epub_path == tmp_path / "demo.epub"
    assert ("initialize", "Demo", "demo") in calls
    assert ("segment", "chapter_001") in calls
    assert ("segment", "chapter_002") in calls


def test_controller_switches_between_library_books(tmp_path):
    first = BookPaths(tmp_path / "first")
    second = BookPaths(tmp_path / "second")
    first.chapter_text("chapter_001").parent.mkdir(parents=True)
    first.chapter_text("chapter_001").write_text("First Book\n", encoding="utf-8")
    second.chapter_text("chapter_001").parent.mkdir(parents=True)
    second.chapter_text("chapter_001").write_text("Second Book\n", encoding="utf-8")
    library_path = tmp_path / "library.json"
    write_json_atomic(
        library_path,
        {
            "books": [
                {
                    "title": "First",
                    "slug": "first",
                    "book_root": str(first.root),
                    "epub_path": str(tmp_path / "first.epub"),
                },
                {
                    "title": "Second",
                    "slug": "second",
                    "book_root": str(second.root),
                    "epub_path": str(tmp_path / "second.epub"),
                },
            ]
        },
    )
    controller = PrototypeUiController(book_root=first.root, library_path=library_path)

    controller.select_book("second")

    assert controller.book_root == second.root
    assert controller.current_book_slug == "second"
    assert controller.chapter_rows()[0].title == "Second Book"


def test_controller_chapter_action_advances_through_pipeline_stages(tmp_path):
    calls = []
    opened = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.sentence_artifact("chapter_001"),
        {"chapter": "chapter_001", "source_path": "chapters/chapter_001.txt", "sentences": []},
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        audio_opener=opened.append,
        fake_tts=True,
    )

    first = controller.run_next_chapter_action("chapter_001")
    second = controller.run_next_chapter_action("chapter_001")
    third = controller.run_next_chapter_action("chapter_001")
    fourth = controller.run_next_chapter_action("chapter_001")

    assert first.stage == ChapterStage.ANNOTATED
    assert second.stage == ChapterStage.SCRIPTED
    assert third.stage == ChapterStage.AUDIO
    assert fourth.stage == ChapterStage.AUDIO
    assert opened == [paths.chapter_audio("chapter_001")]
    assert ("annotate", "chapter_001") in calls
    assert ("build_scripts", "chapter_001") in calls
    assert ("prepare_voices",) in calls
    assert ("synthesize", "chapter_001") in calls
