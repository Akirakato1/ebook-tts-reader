import json

import numpy as np

from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def ensure_voice(self, role_id, voice_record, voice_path):
        return voice_path

    def generate_sentences(self, jobs):
        self.calls.append([dict(job) for job in jobs])
        return [
            GeneratedSentenceAudio(
                sentence_idx=int(job["sentence_idx"]),
                unit_idx=int(job["unit_idx"]),
                role=str(job["role"]),
                speech_type=str(job["type"]),
                samples=np.ones(2400, dtype=np.float32) * 0.05,
                sample_rate=24000,
                voice_config_path=str(job.get("voice_config_path") or ""),
            )
            for job in jobs
        ]


class VariableDurationAdapter(RecordingAdapter):
    def __init__(self, sample_counts):
        super().__init__()
        self.sample_counts = list(sample_counts)

    def generate_sentences(self, jobs):
        self.calls.append([dict(job) for job in jobs])
        result = []
        for job in jobs:
            sample_count = self.sample_counts[int(job["unit_idx"])]
            result.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    unit_idx=int(job["unit_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=np.ones(sample_count, dtype=np.float32) * 0.05,
                    sample_rate=24000,
                    voice_config_path=str(job.get("voice_config_path") or ""),
                )
            )
        return result


def test_session_never_generates_more_than_buffer_limit(tmp_path):
    adapter = RecordingAdapter()
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        target_buffer_seconds=0.2,
        max_buffer_seconds=0.2,
        max_buffer_units=2,
        playback_speed=1.0,
        generation_mode="fast",
    )

    generated = session.fill_buffer(start_unit_id=0)

    assert len(generated) == 2
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 2
    assert session.ready_count == 2


def test_session_logs_realtime_factor_and_cleans_temp_audio(tmp_path):
    adapter = RecordingAdapter()
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=2.0,
        generation_mode="precise",
    )

    session.fill_buffer(start_unit_id=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "session" / "timings.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["unit_ids"] == [0]
    assert rows[0]["playback_speed"] == 2.0
    assert rows[0]["playback_seconds"] == 0.05
    assert rows[0]["realtime_factor"] >= 0
    assert list((tmp_path / "session").glob("*.wav"))

    session.end()

    assert not (tmp_path / "session").exists()


def test_session_can_buffer_audio_without_writing_wav_files(tmp_path):
    adapter = RecordingAdapter()
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=1,
        playback_speed=1.0,
        generation_mode="precise",
        store_audio_files=False,
    )

    [item] = session.fill_buffer(start_unit_id=0)

    assert item.audio_bytes[:4] == b"RIFF"
    assert item.audio_path is None
    assert not list((tmp_path / "session").glob("*.wav"))


def test_session_fills_until_target_buffer_seconds(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2), _unit(3)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=3.0,
        start_buffer_seconds=2.0,
        max_buffer_seconds=4.0,
        max_buffer_units=8,
    )

    generated = session.fill_buffer(start_unit_id=0, min_buffer_seconds=3.0)

    assert [item.unit_id for item in generated] == [0, 1, 2]
    assert session.ready_playback_seconds == 3.0
    assert len(adapter.calls) == 2
    assert [len(call) for call in adapter.calls] == [2, 1]


def test_session_does_not_generate_when_buffer_seconds_are_full(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=2.0,
        start_buffer_seconds=1.0,
        max_buffer_seconds=2.0,
        max_buffer_units=8,
    )

    session.fill_buffer(start_unit_id=0, min_buffer_seconds=2.0)
    second = session.fill_buffer()

    assert second == []
    assert session.ready_playback_seconds == 2.0
    assert len(adapter.calls) == 1


def test_session_tops_up_after_playback_consumes_buffer(tmp_path):
    adapter = VariableDurationAdapter([24000, 24000, 24000, 24000])
    session = ReadAlongSession(
        session_id="s1",
        units=[_unit(0), _unit(1), _unit(2), _unit(3)],
        tts_adapter=adapter,
        session_dir=tmp_path / "session",
        timing_log_path=tmp_path / "session" / "timings.jsonl",
        buffer_limit=2,
        playback_speed=1.0,
        generation_mode="fast",
        target_buffer_seconds=2.0,
        start_buffer_seconds=1.0,
        max_buffer_seconds=3.0,
        max_buffer_units=8,
    )

    session.fill_buffer(start_unit_id=0, min_buffer_seconds=2.0)
    consumed = session.consume_ready()
    generated = session.fill_buffer()

    assert consumed is not None
    assert consumed.unit_id == 0
    assert [item.unit_id for item in generated] == [2]
    assert session.ready_playback_seconds == 2.0
    assert session.ready_unit_ids == [1, 2]


def _unit(unit_id):
    return ReadAlongUnit(
        chapter="chapter_001",
        unit_id=unit_id,
        text=f"Unit {unit_id}.",
        source_start=unit_id * 10,
        source_end=unit_id * 10 + 7,
        role="Narrator",
        role_id="narrator",
        type="narration",
        voice_config_path="voices/narrator.qvp",
    )
