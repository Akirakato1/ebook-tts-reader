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
    assert data["units"] == [
        {"idx": 0, "sentence_idx": 0, "text": "Hello there."},
        {"idx": 1, "sentence_idx": 1, "text": "She waved."},
    ]


def test_sentence_segmenter_splits_dialogue_embedded_narration_into_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        '"Go," Callie said, looking away. "Now." Plain narration.',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: ['"Go," Callie said, looking away. "Now."', "Plain narration."]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": '"Go," Callie said, looking away.'},
        {"idx": 1, "sentence_idx": 0, "text": '"Now."'},
        {"idx": 2, "sentence_idx": 1, "text": "Plain narration."},
    ]


def test_sentence_segmenter_splits_adjacent_quotes_into_role_allocation_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        '"I found this for you." "Wonderful, thank you." Callie took the book.',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: ['"I found this for you." "Wonderful, thank you." Callie took the book.']
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": '"I found this for you."'},
        {"idx": 1, "sentence_idx": 0, "text": '"Wonderful, thank you." Callie took the book.'},
    ]


def test_sentence_segmenter_keeps_trailing_tag_with_quote_context(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        '"Go," Callie said, looking away. "Now."',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: ['"Go," Callie said, looking away. "Now."']
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": '"Go," Callie said, looking away.'},
        {"idx": 1, "sentence_idx": 0, "text": '"Now."'},
    ]


def test_sentence_segmenter_keeps_leading_tag_with_following_quote(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        'Walter said, "I like your jacket." "It is from high school." Callie turned around.',
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            'Walter said, "I like your jacket." "It is from high school." Callie turned around.'
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": 'Walter said, "I like your jacket."'},
        {"idx": 1, "sentence_idx": 0, "text": '"It is from high school." Callie turned around.'},
    ]


def test_sentence_segmenter_handles_smart_quote_role_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        "\u201cWelcome, friend.\u201d Callie smiled. \u201cThank you.\u201d",
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            "\u201cWelcome, friend.\u201d Callie smiled. \u201cThank you.\u201d"
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": "\u201cWelcome, friend.\u201d Callie smiled."},
        {"idx": 1, "sentence_idx": 0, "text": "\u201cThank you.\u201d"},
    ]


def test_sentence_segmenter_preserves_open_quote_continuations_as_role_units(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text(
        "\u201cI could not see your face. You never looked up. "
        "You did what she told you to do.\u201d It was almost a relief.",
        encoding="utf-8",
    )
    segmenter = SentenceSegmenter(
        tokenizer=lambda text: [
            "\u201cI could not see your face.",
            "You never looked up.",
            "You did what she told you to do.\u201d It was almost a relief.",
        ]
    )

    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [unit.to_dict() for unit in artifact.units] == [
        {"idx": 0, "sentence_idx": 0, "text": "\u201cI could not see your face."},
        {"idx": 1, "sentence_idx": 1, "text": "You never looked up."},
        {
            "idx": 2,
            "sentence_idx": 2,
            "text": "You did what she told you to do.\u201d It was almost a relief.",
        },
    ]
