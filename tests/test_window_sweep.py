import json
from pathlib import Path

import numpy as np

from ebook_tts_pipeline.benchmarks.window_sweep import (
    SweepConfig,
    SweepResult,
    SweepRow,
    build_target_windows,
    max_smooth_speed,
    realtime_factor,
    word_count,
    write_outputs,
)
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


def _unit(unit_id, text, role_id="narrator", voice="voices/narrator.qvp"):
    return ReadAlongUnit(
        chapter="chapter_015",
        unit_id=unit_id,
        text=text,
        source_start=unit_id * 10,
        source_end=unit_id * 10 + len(text),
        role="Narrator" if role_id == "narrator" else "Maddy",
        role_id=role_id,
        type="narration" if role_id == "narrator" else "dialogue",
        voice_config_path=voice,
    )


def test_word_count_counts_plain_words():
    assert word_count("One, two three.") == 3


def test_metric_helpers_handle_positive_playback():
    assert realtime_factor(2.0, 4.0) == 0.5
    assert max_smooth_speed(2.0, 4.0) == 2.0


def test_metric_helpers_return_none_when_denominator_is_zero():
    assert realtime_factor(2.0, 0.0) is None
    assert max_smooth_speed(0.0, 4.0) is None


def test_build_target_windows_preserves_boundaries_and_reuses_same_start():
    units = [
        _unit(0, "a" * 70),
        _unit(1, "b" * 40),
        _unit(2, "c" * 55),
        _unit(3, "d" * 80),
        _unit(4, "e" * 25),
    ]
    config = SweepConfig(
        chapter="chapter_015",
        start_chars=100,
        step_chars=100,
        max_vram_gb=10.0,
        playback_speed=1.0,
        max_targets=3,
    )

    windows = build_target_windows(units, config)

    assert [(window.target_chars, [unit.unit_id for unit in window.units]) for window in windows] == [
        (100, [0]),
        (200, [0, 1, 2]),
        (300, [0, 1, 2, 3, 4]),
    ]
    assert windows[0].actual_chars == 70
    assert windows[0].word_count == 1
    assert windows[0].role_ids == ["narrator"]


def test_build_target_windows_reuses_same_start_for_independent_targets():
    units = [
        _unit(0, "warmup " * 20),
        _unit(1, "alpha " * 20),
        _unit(2, "bravo " * 20),
        _unit(3, "charlie " * 20),
    ]

    windows = build_target_windows(
        units,
        SweepConfig(chapter="chapter_015", start_chars=100, step_chars=100, max_vram_gb=10.0, max_targets=2),
    )

    assert windows[0].unit_ids[0] == 0
    assert windows[1].unit_ids[0] == 0


def test_build_target_windows_skips_duplicate_unit_sets():
    units = [
        _unit(0, "a" * 70),
        _unit(1, "b" * 400),
        _unit(2, "c" * 20),
    ]

    windows = build_target_windows(
        units,
        SweepConfig(chapter="chapter_015", start_chars=100, step_chars=100, max_vram_gb=10.0),
    )

    assert [(window.target_chars, window.actual_chars, window.unit_ids) for window in windows] == [
        (100, 70, [0]),
        (500, 490, [0, 1, 2]),
    ]


def test_sweep_config_defaults_to_three_repeats():
    config = SweepConfig(chapter="chapter_015")

    assert config.repeat_count == 3
    assert config.warmup_text == "Test"


class FixedAudioAdapter:
    def __init__(self):
        self.calls = []
        self.closed = False

    def ensure_voice(self, role_id, voice_record, voice_path):
        return voice_path

    def close(self):
        self.closed = True

    def generate_sentences(self, jobs):
        self.calls.append([dict(job) for job in jobs])
        result = []
        for job in jobs:
            samples = np.ones(24000, dtype=np.float32)
            result.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    unit_idx=int(job["unit_idx"]),
                    role=str(job["role"]),
                    speech_type=str(job["type"]),
                    samples=samples,
                    sample_rate=24000,
                    voice_config_path=str(job.get("voice_config_path") or ""),
                )
            )
        return result


def test_run_sweep_warms_backend_then_averages_repeats_and_stops_at_vram_limit(tmp_path):
    from ebook_tts_pipeline.benchmarks.window_sweep import run_sweep

    units = [
        _unit(0, "a" * 70),
        _unit(1, "b" * 40),
        _unit(2, "c" * 55),
        _unit(3, "d" * 80),
    ]
    perf_events = [
        {
            "cuda_after": {
                "max_memory_reserved": 15 * 1024**3,
                "max_memory_allocated": 14 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 2 * 1024**3,
            },
            "text_char_max": 4,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 12 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 4 * 1024**3,
                "max_memory_allocated": 3 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 9 * 1024**3,
            },
            "text_char_max": 120,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 10 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 5 * 1024**3,
                "max_memory_allocated": 3 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 8 * 1024**3,
            },
            "text_char_max": 120,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 11 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 6 * 1024**3,
                "max_memory_allocated": 3 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 7 * 1024**3,
            },
            "text_char_max": 120,
        },
        {
            "cuda_after": {
                "max_memory_reserved": 15 * 1024**3,
                "max_memory_allocated": 14 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 2 * 1024**3,
            },
            "text_char_max": 4,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 7 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 9 * 1024**3,
                "max_memory_allocated": 6 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 4 * 1024**3,
            },
            "text_char_max": 260,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 8 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 9 * 1024**3,
                "max_memory_allocated": 6 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 5 * 1024**3,
            },
            "text_char_max": 260,
        },
        {
            "cuda_before": {"mem_total": 16 * 1024**3, "mem_free": 9 * 1024**3},
            "cuda_after": {
                "max_memory_reserved": 8 * 1024**3,
                "max_memory_allocated": 6 * 1024**3,
                "mem_total": 16 * 1024**3,
                "mem_free": 6 * 1024**3,
            },
            "text_char_max": 260,
        },
    ]
    adapters = []

    def adapter_factory():
        adapter = FixedAudioAdapter()
        adapters.append(adapter)
        return adapter

    result = run_sweep(
        units=units,
        adapter_factory=adapter_factory,
        config=SweepConfig(
            chapter="chapter_015",
            start_chars=100,
            step_chars=100,
            max_vram_gb=10.0,
            playback_speed=1.0,
            repeat_count=3,
            max_targets=3,
        ),
        machine_profile={"gpu_name": "test gpu"},
        perf_event_reader=lambda: perf_events.pop(0),
    )

    assert [row.target_chars for row in result.rows] == [100, 200]
    assert len(adapters) == 2
    assert all(len(adapter.calls) == 4 for adapter in adapters)
    assert all(adapter.calls[0][0]["text"] == "Test" for adapter in adapters)
    assert all(adapter.closed for adapter in adapters)
    assert all(row.repeat_count == 3 for row in result.rows)
    assert result.rows[0].rtf_at_1x is not None
    assert result.rows[0].max_smooth_speed is not None
    assert result.rows[0].device_vram_used_before_gb == 6.0
    assert result.rows[0].device_vram_used_after_gb == 9.0
    assert result.rows[1].peak_vram_reserved_gb == 9.0
    assert result.rows[1].device_vram_used_after_gb == 12.0
    assert result.stop_reason == "vram_limit"


def test_perf_log_reader_returns_new_json_rows(tmp_path):
    from ebook_tts_pipeline.benchmarks.window_sweep import PerfLogReader

    path = tmp_path / "perf.jsonl"
    path.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    reader = PerfLogReader(path)

    assert reader.read_next() == {"step": 1}

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 2}) + "\n")

    assert reader.read_next() == {"step": 2}


def test_write_outputs_creates_json_csv_and_markdown(tmp_path):
    result = SweepResult(
        config=SweepConfig(chapter="chapter_015", start_chars=100, step_chars=100, max_vram_gb=10.0),
        machine_profile={"gpu_name": "RTX Test", "gpu_total_vram_gb": 16.0},
        rows=[
            SweepRow(
                target_chars=100,
                actual_chars=120,
                word_count=20,
                unit_count=2,
                unit_ids=[1, 2],
                role_ids=["narrator"],
                voice_config_paths=["voices/narrator.qvp"],
                generation_seconds=4.0,
                playback_seconds=8.0,
                rtf_at_1x=0.5,
                max_smooth_speed=2.0,
                peak_vram_reserved_gb=5.0,
                peak_vram_allocated_gb=4.5,
                device_vram_used_before_gb=5.5,
                device_vram_used_after_gb=6.5,
                text_char_max=120,
                audio_seconds=[8.0],
                repeat_count=3,
                success=True,
            )
        ],
        stop_reason="chapter_exhausted",
    )

    paths = write_outputs(
        result=result,
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
        date_slug="2026-06-23",
    )

    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    assert "rtf_at_1x" in paths["csv"].read_text(encoding="utf-8")
    assert "device_vram_used_after_gb" in paths["csv"].read_text(encoding="utf-8")
    assert "RTX Test" in paths["markdown"].read_text(encoding="utf-8")
    assert "0.500" in paths["markdown"].read_text(encoding="utf-8")
