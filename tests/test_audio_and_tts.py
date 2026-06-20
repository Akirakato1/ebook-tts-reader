import wave

from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


def test_fake_tts_and_audio_builder_write_sentence_timeline(tmp_path):
    adapter = FakeTtsAdapter(sample_rate=1000, samples_per_character=10)
    builder = ChapterAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=100)
    jobs = [
        {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "Hello."},
        {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Hi."},
    ]

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=jobs,
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert result["sentences"][0]["start_ms"] == 0
    assert result["sentences"][0]["end_ms"] == 60
    assert result["sentences"][1]["start_ms"] == 160
    assert result["sentences"][1]["end_ms"] == 190

    with wave.open(str(tmp_path / "chapter_001.wav"), "rb") as wav:
        assert wav.getframerate() == 1000
        assert wav.getnchannels() == 1
