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


def test_audio_builder_timeline_preserves_unit_indices(tmp_path):
    adapter = FakeTtsAdapter(sample_rate=1000, samples_per_character=10)
    builder = ChapterAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=0)
    jobs = [
        {"sentence_idx": 0, "unit_idx": 0, "role": "Elena", "type": "dialogue", "text": '"Hi,"'},
        {"sentence_idx": 0, "unit_idx": 1, "role": "Narrator", "type": "narration", "text": "she said."},
    ]

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=jobs,
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert [(item["sentence_idx"], item["unit_idx"], item["role"]) for item in result["sentences"]] == [
        (0, 0, "Elena"),
        (0, 1, "Narrator"),
    ]


def test_audio_builder_uses_short_pause_between_units_in_same_sentence(tmp_path):
    adapter = FakeTtsAdapter(sample_rate=1000, samples_per_character=10)
    builder = ChapterAudioBuilder(
        tts_adapter=adapter,
        pause_between_sentences_ms=100,
        intra_sentence_pause_ms=20,
    )
    jobs = [
        {"sentence_idx": 0, "unit_idx": 0, "role": "Elena", "type": "dialogue", "text": '"Hello"'},
        {"sentence_idx": 0, "unit_idx": 1, "role": "Narrator", "type": "narration", "text": "she said"},
        {"sentence_idx": 1, "unit_idx": 2, "role": "Narrator", "type": "narration", "text": "Then left"},
    ]

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=jobs,
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    sentences = result["sentences"]
    assert sentences[1]["start_ms"] - sentences[0]["end_ms"] == 20
    assert sentences[2]["start_ms"] - sentences[1]["end_ms"] == 100


class UnitStreamingAdapter:
    def __init__(self):
        self.calls = []

    def generate_sentences(self, jobs):
        raise AssertionError("builder should consume streaming batches when available")

    def generate_sentence_batches(self, jobs):
        self.calls.append([job["unit_idx"] for job in jobs])
        for job in jobs:
            yield [
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    unit_idx=int(job["unit_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=np.ones(10, dtype=np.float32),
                    sample_rate=1000,
                )
            ]


def test_audio_builder_uses_unit_pause_across_generation_window_boundaries(tmp_path):
    adapter = UnitStreamingAdapter()
    builder = ChapterAudioBuilder(
        tts_adapter=adapter,
        pause_between_sentences_ms=100,
        intra_sentence_pause_ms=20,
    )

    result = builder.build_chapter_audio_from_windows(
        chapter="chapter_001",
        job_windows=[
            [{"sentence_idx": 0, "unit_idx": 0, "role": "Elena", "type": "dialogue", "text": '"Hello"'}],
            [{"sentence_idx": 0, "unit_idx": 1, "role": "Narrator", "type": "narration", "text": "she said"}],
            [{"sentence_idx": 1, "unit_idx": 2, "role": "Narrator", "type": "narration", "text": "Then left"}],
        ],
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert adapter.calls == [[0], [1], [2]]
    assert [item["start_ms"] for item in result["sentences"]] == [0, 30, 140]
    assert [item["end_ms"] for item in result["sentences"]] == [10, 40, 150]


class BlockPauseAdapter:
    def generate_sentences(self, jobs):
        raise AssertionError("builder should consume streaming batches when available")

    def generate_sentence_batches(self, jobs):
        yield [
            GeneratedSentenceAudio(
                sentence_idx=0,
                unit_idx=0,
                role="Narrator",
                speech_type="narration",
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
                pause_after_ms=0,
            ),
            GeneratedSentenceAudio(
                sentence_idx=1,
                unit_idx=1,
                role="Narrator",
                speech_type="narration",
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
            ),
            GeneratedSentenceAudio(
                sentence_idx=2,
                unit_idx=2,
                role="Elena",
                speech_type="dialogue",
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
            ),
        ]


def test_audio_builder_honors_adapter_pause_override_for_generated_blocks(tmp_path):
    builder = ChapterAudioBuilder(
        tts_adapter=BlockPauseAdapter(),
        pause_between_sentences_ms=100,
        intra_sentence_pause_ms=20,
    )

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=[
            {"sentence_idx": 0, "unit_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "unit_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
            {"sentence_idx": 2, "unit_idx": 2, "role": "Elena", "type": "dialogue", "text": "Hi."},
        ],
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    sentences = result["sentences"]
    assert sentences[1]["start_ms"] - sentences[0]["end_ms"] == 0
    assert sentences[2]["start_ms"] - sentences[1]["end_ms"] == 100


def test_audio_builder_applies_tts_speed_to_generated_samples(tmp_path):
    adapter = FakeTtsAdapter(sample_rate=1000, samples_per_character=10)
    builder = ChapterAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=0, tts_speed=2.0)

    result = builder.build_chapter_audio(
        chapter="chapter_001",
        jobs=[{"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "123456"}],
        audio_path=tmp_path / "chapter_001.wav",
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert result["sentences"][0]["end_ms"] == 30
    with wave.open(str(tmp_path / "chapter_001.wav"), "rb") as wav:
        assert wav.getnframes() == 30


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


class StreamingBatchAdapter:
    def __init__(self):
        self.calls = []

    def generate_sentences(self, jobs):
        raise AssertionError("builder should consume streaming batches when available")

    def generate_sentence_batches(self, jobs):
        self.calls.append([job["sentence_idx"] for job in jobs])
        yield [
            GeneratedSentenceAudio(
                sentence_idx=int(jobs[0]["sentence_idx"]),
                role=str(jobs[0]["role"]),
                speech_type=str(jobs[0]["type"]),
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
            )
        ]
        yield [
            GeneratedSentenceAudio(
                sentence_idx=int(jobs[1]["sentence_idx"]),
                role=str(jobs[1]["role"]),
                speech_type=str(jobs[1]["type"]),
                samples=np.ones(20, dtype=np.float32),
                sample_rate=1000,
            )
        ]


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


def test_windowed_audio_builder_spools_each_streamed_tts_batch(tmp_path):
    adapter = StreamingBatchAdapter()
    builder = RecordingAudioBuilder(tts_adapter=adapter, pause_between_sentences_ms=0)
    audio_path = tmp_path / "chapter_001.wav"

    result = builder.build_chapter_audio_from_windows(
        chapter="chapter_001",
        job_windows=[
            [
                {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
                {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
            ]
        ],
        audio_path=audio_path,
        timeline_path=tmp_path / "chapter_001.timeline.json",
    )

    assert adapter.calls == [[0, 1]]
    assert builder.wav_writes == ["00000.wav", "00001.wav"]
    assert [sentence["start_ms"] for sentence in result["sentences"]] == [0, 10]
    assert [sentence["end_ms"] for sentence in result["sentences"]] == [10, 30]
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getnframes() == 30
    assert not any(tmp_path.glob("chapter_001.chunks*"))
