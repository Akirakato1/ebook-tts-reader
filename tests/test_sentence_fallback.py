from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.paths import BookPaths


def test_sentence_segmenter_falls_back_when_nltk_data_is_missing(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Hello there. She waved. Stay close!", encoding="utf-8")

    segmenter = SentenceSegmenter(tokenizer=None, allow_nltk_download=False)
    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [sentence.text for sentence in artifact.sentences] == [
        "Hello there.",
        "She waved.",
        "Stay close!",
    ]
