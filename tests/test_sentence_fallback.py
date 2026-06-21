import sys

from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.paths import BookPaths


class MissingNltkData:
    __version__ = "test"

    def sent_tokenize(self, text):
        raise LookupError("punkt missing")

    def download(self, *args, **kwargs):
        raise AssertionError("download should not run when allow_nltk_download is false")


def test_sentence_segmenter_falls_back_when_nltk_data_is_missing(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "demo")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Hello there. She waved. Stay close!", encoding="utf-8")

    monkeypatch.delitem(sys.modules, "nltk", raising=False)
    monkeypatch.delitem(sys.modules, "nltk.tokenize", raising=False)
    monkeypatch.setitem(sys.modules, "nltk", MissingNltkData())
    segmenter = SentenceSegmenter(tokenizer=None, allow_nltk_download=False)
    artifact = segmenter.segment_chapter(paths, "chapter_001")

    assert [sentence.text for sentence in artifact.sentences] == [
        "Hello there.",
        "She waved.",
        "Stay close!",
    ]
