from ebook_tts_pipeline.domain import SentenceArtifact
from ebook_tts_pipeline.ingestion import ChapterSplitter, SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json
from ebook_tts_pipeline.paths import BookPaths


def test_chapter_splitter_writes_confident_chapters(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.source_book.parent.mkdir(parents=True)
    paths.source_book.write_text(
        "Chapter 1\nThe first room was silent.\n\n"
        "Chapter 2\nThe second room was loud.\n",
        encoding="utf-8",
    )

    result = ChapterSplitter().split_source_book(paths)

    assert result.chapters == ["chapter_001", "chapter_002"]
    assert paths.chapter_text("chapter_001").read_text(encoding="utf-8").startswith("The first")
    assert paths.chapter_text("chapter_002").read_text(encoding="utf-8").startswith("The second")


def test_chapter_splitter_rejects_low_confidence_source(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.source_book.parent.mkdir(parents=True)
    paths.source_book.write_text("A book with no clear headings.", encoding="utf-8")

    result = ChapterSplitter().split_source_book(paths)

    assert result.chapters == []
    assert result.reason == "low_confidence_chapter_split"


def test_sentence_segmenter_writes_canonical_artifact(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Hello there. She waved.", encoding="utf-8")
    segmenter = SentenceSegmenter(tokenizer=lambda text: ["Hello there.", "She waved."])

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert isinstance(artifact, SentenceArtifact)
    data = read_json(paths.sentence_artifact("chapter_001"))
    assert data["sentences"] == [
        {"idx": 0, "text": "Hello there."},
        {"idx": 1, "text": "She waved."},
    ]
