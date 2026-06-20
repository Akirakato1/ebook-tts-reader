import json

from ebook_tts_pipeline.domain import Sentence, SentenceArtifact
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


def test_book_paths_match_spec_layout(tmp_path):
    paths = BookPaths(tmp_path / "books" / "demo")

    assert paths.registry.name == "registry.json"
    assert paths.source_book.as_posix().endswith("source/book.txt")
    assert paths.chapter_text("chapter_001").as_posix().endswith("chapters/chapter_001.txt")
    assert paths.sentence_artifact("chapter_001").as_posix().endswith(
        "sentence_segments/chapter_001.sentences.json"
    )
    assert paths.annotation("chapter_001").as_posix().endswith(
        "annotations/chapter_001.annotation.json"
    )
    assert paths.chapter_audio("chapter_001").as_posix().endswith("audio/chapter_001.wav")
    assert paths.chapter_timeline("chapter_001").as_posix().endswith(
        "audio/chapter_001.timeline.json"
    )
    assert paths.voice_qvp("elena").as_posix().endswith("voices/elena.qvp")


def test_sentence_artifact_serializes_with_stable_sentence_indexes():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test", "language": "english", "version": "1"},
        sentences=[Sentence(idx=0, text="Hello."), Sentence(idx=1, text="Goodbye.")],
    )

    data = artifact.to_dict()
    restored = SentenceArtifact.from_dict(data)

    assert [s.idx for s in restored.sentences] == [0, 1]
    assert restored.sentences[1].text == "Goodbye."


def test_write_json_atomic_creates_parent_and_valid_json(tmp_path):
    path = tmp_path / "nested" / "data.json"

    write_json_atomic(path, {"ok": True})

    assert read_json(path) == {"ok": True}
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
