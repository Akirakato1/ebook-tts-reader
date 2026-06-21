import wave

import numpy as np

from ebook_tts_pipeline.audio import ChapterAudioBuilder
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
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


class RecordingWindowAdapter:
    def __init__(self):
        self.calls = []

    def generate_sentences(self, jobs):
        self.calls.append([job["sentence_idx"] for job in jobs])
        return [
            GeneratedSentenceAudio(
                sentence_idx=int(job["sentence_idx"]),
                role=str(job["role"]),
                speech_type=str(job["type"]),
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
            )
            for job in jobs
        ]


class RecordingAudioBuilder(ChapterAudioBuilder):
    def __init__(self, tts_adapter, pause_between_sentences_ms):
        super().__init__(tts_adapter=tts_adapter, pause_between_sentences_ms=pause_between_sentences_ms)
        self.wav_writes = []

    def _write_wav(self, path, samples, sample_rate):
        self.wav_writes.append(path.name)
        super()._write_wav(path, samples, sample_rate)


def test_windowed_audio_builder_spools_chunks_and_removes_temporary_files(tmp_path):
    adapter = RecordingWindowAdapter()
    builder = RecordingAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=0)
    audio_path = tmp_path / "chapter_001.wav"

    result = builder.build_chapter_audio_from_windows(
        chapter="chapter_001",
        job_windows=[
            [{"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."}],
            [{"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."}],
        ],
        audio_path=audio_path,
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert adapter.calls == [[0], [1]]
    assert builder.wav_writes == ["00000.wav", "00001.wav"]
    assert [sentence["start_ms"] for sentence in result["sentences"]] == [0, 10]
    assert [sentence["end_ms"] for sentence in result["sentences"]] == [10, 20]
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getnframes() == 20
        assert wav.getframerate() == 1000
    assert not any(tmp_path.glob("chapter_001.chunks*"))
