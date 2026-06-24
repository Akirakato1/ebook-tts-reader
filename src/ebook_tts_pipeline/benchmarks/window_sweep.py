from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.read_along.units import ReadAlongUnit
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio, TtsAdapter


@dataclass(frozen=True)
class SweepConfig:
    chapter: str
    start_chars: int = 100
    step_chars: int = 100
    max_vram_gb: float = 10.0
    playback_speed: float = 1.0
    generation_mode: str = "balanced"
    warmup_text: str = "Test"
    repeat_count: int = 3
    max_targets: Optional[int] = None


@dataclass(frozen=True)
class WindowSpec:
    target_chars: int
    units: List[ReadAlongUnit]
    actual_chars: int
    word_count: int
    role_ids: List[str]
    voice_config_paths: List[str]

    @property
    def unit_ids(self) -> List[int]:
        return [unit.unit_id for unit in self.units]

    def to_jobs(self) -> List[Dict[str, Any]]:
        return [unit.to_tts_job() for unit in self.units]


@dataclass(frozen=True)
class SweepRow:
    target_chars: int
    actual_chars: int
    word_count: int
    unit_count: int
    unit_ids: List[int]
    role_ids: List[str]
    voice_config_paths: List[str]
    generation_seconds: float
    playback_seconds: float
    rtf_at_1x: Optional[float]
    max_smooth_speed: Optional[float]
    peak_vram_reserved_gb: Optional[float]
    peak_vram_allocated_gb: Optional[float]
    device_vram_used_before_gb: Optional[float]
    device_vram_used_after_gb: Optional[float]
    text_char_max: Optional[int]
    audio_seconds: List[float]
    repeat_count: int
    success: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    machine_profile: Dict[str, Any]
    rows: List[SweepRow]
    stop_reason: str


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def realtime_factor(generation_seconds: float, playback_seconds: float) -> Optional[float]:
    if playback_seconds <= 0:
        return None
    return generation_seconds / playback_seconds


def max_smooth_speed(generation_seconds: float, playback_seconds: float) -> Optional[float]:
    if generation_seconds <= 0:
        return None
    return playback_seconds / generation_seconds


def build_target_windows(
    units: Sequence[ReadAlongUnit],
    config: SweepConfig,
) -> List[WindowSpec]:
    unit_list = list(units)
    windows: List[WindowSpec] = []
    target = int(config.start_chars)
    step = max(1, int(config.step_chars))
    last_unit_ids: Optional[Tuple[int, ...]] = None
    while unit_list:
        selected = _take_window_units(unit_list, start_index=0, target_chars=target)
        if not selected:
            break
        selected_ids = tuple(unit.unit_id for unit in selected)
        if selected_ids != last_unit_ids:
            windows.append(_window_spec(target_chars=target, units=selected))
            last_unit_ids = selected_ids
            if config.max_targets is not None and len(windows) >= int(config.max_targets):
                break
            if len(selected) >= len(unit_list):
                break
        target += step
    return windows


def run_sweep(
    units: Sequence[ReadAlongUnit],
    adapter_factory: Callable[[], TtsAdapter],
    config: SweepConfig,
    machine_profile: Dict[str, Any],
    perf_event_reader: Callable[[], Dict[str, Any]],
) -> SweepResult:
    windows = build_target_windows(units, config)
    stop_reason = "chapter_exhausted"
    rows: List[SweepRow] = []
    for window in windows:
        adapter = adapter_factory()
        repeats: List[SweepRow] = []
        try:
            warmup_text = str(config.warmup_text or "").strip()
            if warmup_text:
                warmup_row = _generate_window_row(
                    adapter=adapter,
                    window=_warmup_window(window, warmup_text),
                    playback_speed=config.playback_speed,
                    perf_event_reader=perf_event_reader,
                )
                if not warmup_row.success:
                    rows.append(_warmup_failure_row(window, warmup_row))
                    stop_reason = "generation_error"
                    break
            for _ in range(max(1, int(config.repeat_count))):
                repeats.append(
                    _generate_window_row(
                        adapter=adapter,
                        window=window,
                        playback_speed=config.playback_speed,
                        perf_event_reader=perf_event_reader,
                    )
                )
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        row = _average_repeat_rows(window, repeats)
        rows.append(row)
        if not row.success:
            stop_reason = "generation_error"
            break
        peak_reserved = row.peak_vram_reserved_gb or 0.0
        peak_allocated = row.peak_vram_allocated_gb or 0.0
        peak_device = max(
            row.device_vram_used_before_gb or 0.0,
            row.device_vram_used_after_gb or 0.0,
        )
        if max(peak_reserved, peak_allocated, peak_device) >= float(config.max_vram_gb):
            stop_reason = "vram_limit"
            break
    return SweepResult(
        config=config,
        machine_profile=dict(machine_profile),
        rows=rows,
        stop_reason=stop_reason,
    )


class PerfLogReader:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._offset = 0

    def read_next(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            line = handle.readline()
            self._offset = handle.tell()
        if not line:
            return {}
        return dict(json.loads(line))


def collect_machine_profile(config: PipelineConfig) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
        "tts_backend": config.tts_backend,
        "wsl_distro": config.wsl_distro,
        "wsl_python": config.wsl_python,
        "qwen_model_choice": config.qwen_model_choice,
        "qwen_model_root": config.qwen_model_root,
        "qwen_precision": config.qwen_precision,
        "qwen_attention": config.qwen_attention,
    }
    if config.tts_backend == "wsl":
        profile.update(_collect_wsl_profile(config))
    return profile


def write_outputs(
    result: SweepResult,
    logs_dir: Path | str,
    docs_dir: Path | str,
    date_slug: str,
) -> Dict[str, Path]:
    logs = Path(logs_dir)
    docs = Path(docs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    base_name = f"window_sweep_{result.config.chapter}"
    json_path = logs / f"{base_name}.json"
    csv_path = logs / f"{base_name}.csv"
    markdown_path = docs / f"{date_slug}-{result.config.chapter.replace('_', '-')}-window-sweep.md"
    json_path.write_text(json.dumps(_result_payload(result), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, result.rows)
    markdown_path.write_text(_markdown_summary(result), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _take_window_units(
    units: Sequence[ReadAlongUnit],
    start_index: int,
    target_chars: int,
) -> List[ReadAlongUnit]:
    selected: List[ReadAlongUnit] = []
    total = 0
    for unit in list(units)[start_index:]:
        unit_chars = len(unit.text)
        if selected and total + unit_chars > int(target_chars):
            break
        selected.append(unit)
        total += unit_chars
        if total >= int(target_chars):
            break
    return selected


def _window_spec(target_chars: int, units: Sequence[ReadAlongUnit]) -> WindowSpec:
    unit_list = list(units)
    text = " ".join(unit.text for unit in unit_list)
    return WindowSpec(
        target_chars=int(target_chars),
        units=unit_list,
        actual_chars=sum(len(unit.text) for unit in unit_list),
        word_count=word_count(text),
        role_ids=sorted({unit.role_id for unit in unit_list}),
        voice_config_paths=sorted({str(unit.voice_config_path or "") for unit in unit_list if unit.voice_config_path}),
    )


def _warmup_window(window: WindowSpec, text: str) -> WindowSpec:
    first = window.units[0]
    warmup_unit = replace(first, text=text)
    return _window_spec(target_chars=len(text), units=[warmup_unit])


def _warmup_failure_row(window: WindowSpec, warmup_row: SweepRow) -> SweepRow:
    return SweepRow(
        target_chars=window.target_chars,
        actual_chars=window.actual_chars,
        word_count=window.word_count,
        unit_count=len(window.units),
        unit_ids=window.unit_ids,
        role_ids=window.role_ids,
        voice_config_paths=window.voice_config_paths,
        generation_seconds=warmup_row.generation_seconds,
        playback_seconds=0.0,
        rtf_at_1x=None,
        max_smooth_speed=None,
        peak_vram_reserved_gb=warmup_row.peak_vram_reserved_gb,
        peak_vram_allocated_gb=warmup_row.peak_vram_allocated_gb,
        device_vram_used_before_gb=warmup_row.device_vram_used_before_gb,
        device_vram_used_after_gb=warmup_row.device_vram_used_after_gb,
        text_char_max=warmup_row.text_char_max,
        audio_seconds=[],
        repeat_count=0,
        success=False,
        error=f"warmup failed: {warmup_row.error}",
    )


def _generate_window_row(
    adapter: TtsAdapter,
    window: WindowSpec,
    playback_speed: float,
    perf_event_reader: Callable[[], Dict[str, Any]],
) -> SweepRow:
    started = time.perf_counter()
    try:
        generated = adapter.generate_sentences(window.to_jobs())
        generation_seconds = time.perf_counter() - started
        playback_seconds = _playback_seconds(generated, playback_speed)
        perf_event = perf_event_reader() or {}
        peak_reserved, peak_allocated = _vram_peaks_gb(perf_event)
        device_before = _device_vram_used_gb(perf_event, "cuda_before")
        device_after = _device_vram_used_gb(perf_event, "cuda_after")
        audio_seconds = [
            len(item.samples) / item.sample_rate
            for item in generated
            if item.sample_rate
        ]
        return SweepRow(
            target_chars=window.target_chars,
            actual_chars=window.actual_chars,
            word_count=window.word_count,
            unit_count=len(window.units),
            unit_ids=window.unit_ids,
            role_ids=window.role_ids,
            voice_config_paths=window.voice_config_paths,
            generation_seconds=generation_seconds,
            playback_seconds=playback_seconds,
            rtf_at_1x=realtime_factor(generation_seconds, playback_seconds),
            max_smooth_speed=max_smooth_speed(generation_seconds, playback_seconds),
            peak_vram_reserved_gb=peak_reserved,
            peak_vram_allocated_gb=peak_allocated,
            device_vram_used_before_gb=device_before,
            device_vram_used_after_gb=device_after,
            text_char_max=_optional_int(perf_event.get("text_char_max")),
            audio_seconds=audio_seconds,
            repeat_count=1,
            success=True,
            error=None,
        )
    except Exception as exc:
        generation_seconds = time.perf_counter() - started
        return SweepRow(
            target_chars=window.target_chars,
            actual_chars=window.actual_chars,
            word_count=window.word_count,
            unit_count=len(window.units),
            unit_ids=window.unit_ids,
            role_ids=window.role_ids,
            voice_config_paths=window.voice_config_paths,
            generation_seconds=generation_seconds,
            playback_seconds=0.0,
            rtf_at_1x=None,
            max_smooth_speed=None,
            peak_vram_reserved_gb=None,
            peak_vram_allocated_gb=None,
            device_vram_used_before_gb=None,
            device_vram_used_after_gb=None,
            text_char_max=None,
            audio_seconds=[],
            repeat_count=1,
            success=False,
            error=str(exc),
        )


def _average_repeat_rows(window: WindowSpec, rows: Sequence[SweepRow]) -> SweepRow:
    row_list = list(rows)
    successful = [row for row in row_list if row.success]
    if len(successful) != len(row_list):
        errors = "; ".join(str(row.error) for row in row_list if row.error)
        return SweepRow(
            target_chars=window.target_chars,
            actual_chars=window.actual_chars,
            word_count=window.word_count,
            unit_count=len(window.units),
            unit_ids=window.unit_ids,
            role_ids=window.role_ids,
            voice_config_paths=window.voice_config_paths,
            generation_seconds=_avg([row.generation_seconds for row in row_list]),
            playback_seconds=_avg([row.playback_seconds for row in successful]),
            rtf_at_1x=None,
            max_smooth_speed=None,
            peak_vram_reserved_gb=_max_optional([row.peak_vram_reserved_gb for row in row_list]),
            peak_vram_allocated_gb=_max_optional([row.peak_vram_allocated_gb for row in row_list]),
            device_vram_used_before_gb=_max_optional([row.device_vram_used_before_gb for row in row_list]),
            device_vram_used_after_gb=_max_optional([row.device_vram_used_after_gb for row in row_list]),
            text_char_max=_max_optional_int([row.text_char_max for row in row_list]),
            audio_seconds=[],
            repeat_count=len(row_list),
            success=False,
            error=errors or "repeat failed",
        )
    generation_seconds = _avg([row.generation_seconds for row in successful])
    playback_seconds = _avg([row.playback_seconds for row in successful])
    return SweepRow(
        target_chars=window.target_chars,
        actual_chars=window.actual_chars,
        word_count=window.word_count,
        unit_count=len(window.units),
        unit_ids=window.unit_ids,
        role_ids=window.role_ids,
        voice_config_paths=window.voice_config_paths,
        generation_seconds=generation_seconds,
        playback_seconds=playback_seconds,
        rtf_at_1x=realtime_factor(generation_seconds, playback_seconds),
        max_smooth_speed=max_smooth_speed(generation_seconds, playback_seconds),
        peak_vram_reserved_gb=_max_optional([row.peak_vram_reserved_gb for row in successful]),
        peak_vram_allocated_gb=_max_optional([row.peak_vram_allocated_gb for row in successful]),
        device_vram_used_before_gb=_max_optional([row.device_vram_used_before_gb for row in successful]),
        device_vram_used_after_gb=_max_optional([row.device_vram_used_after_gb for row in successful]),
        text_char_max=_max_optional_int([row.text_char_max for row in successful]),
        audio_seconds=[_avg([sum(row.audio_seconds) for row in successful])],
        repeat_count=len(successful),
        success=True,
        error=None,
    )


def _playback_seconds(items: Sequence[GeneratedSentenceAudio], playback_speed: float) -> float:
    raw_seconds = sum(len(item.samples) / item.sample_rate for item in items if item.sample_rate)
    return raw_seconds / max(0.1, float(playback_speed))


def _vram_peaks_gb(perf_event: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    cuda_after = perf_event.get("cuda_after")
    if not isinstance(cuda_after, dict):
        return None, None
    reserved = _bytes_to_gb(
        max(
            int(cuda_after.get("max_memory_reserved", 0) or 0),
            int(cuda_after.get("memory_reserved", 0) or 0),
        )
    )
    allocated = _bytes_to_gb(
        max(
            int(cuda_after.get("max_memory_allocated", 0) or 0),
            int(cuda_after.get("memory_allocated", 0) or 0),
        )
    )
    return reserved, allocated


def _device_vram_used_gb(perf_event: Dict[str, Any], snapshot_key: str) -> Optional[float]:
    snapshot = perf_event.get(snapshot_key)
    if not isinstance(snapshot, dict):
        return None
    total = int(snapshot.get("mem_total", 0) or 0)
    free = int(snapshot.get("mem_free", 0) or 0)
    if total <= 0 or free < 0:
        return None
    return _bytes_to_gb(max(0, total - free))


def _collect_wsl_profile(config: PipelineConfig) -> Dict[str, Any]:
    code = (
        "import json, torch, flash_attn; "
        "payload={"
        "'torch_version': torch.__version__, "
        "'cuda_version': torch.version.cuda, "
        "'flash_attn_version': flash_attn.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'gpu_total_vram_gb': round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 3) if torch.cuda.is_available() else None"
        "}; "
        "print(json.dumps(payload, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "-d",
                config.wsl_distro,
                "-u",
                "root",
                "--",
                config.wsl_python,
                "-c",
                code,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return dict(json.loads(completed.stdout.strip()))
    except Exception as exc:
        return {"profile_error": str(exc)}


def _result_payload(result: SweepResult) -> Dict[str, Any]:
    return {
        "config": asdict(result.config),
        "machine_profile": dict(result.machine_profile),
        "stop_reason": result.stop_reason,
        "rows": [asdict(row) for row in result.rows],
    }


def _write_csv(path: Path, rows: Sequence[SweepRow]) -> None:
    fieldnames = [
        "target_chars",
        "actual_chars",
        "word_count",
        "unit_count",
        "unit_ids",
        "role_ids",
        "voice_config_paths",
        "generation_seconds",
        "playback_seconds",
        "rtf_at_1x",
        "max_smooth_speed",
        "peak_vram_reserved_gb",
        "peak_vram_allocated_gb",
        "device_vram_used_before_gb",
        "device_vram_used_after_gb",
        "text_char_max",
        "audio_seconds",
        "repeat_count",
        "success",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            for key in ("unit_ids", "role_ids", "voice_config_paths", "audio_seconds"):
                payload[key] = json.dumps(payload[key], ensure_ascii=False)
            writer.writerow(payload)


def _markdown_summary(result: SweepResult) -> str:
    lines = [
        f"# {result.config.chapter} Qwen Window Sweep",
        "",
        "## Machine Profile",
        "",
    ]
    for key, value in sorted(result.machine_profile.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            f"- Stop reason: `{result.stop_reason}`",
            f"- Repeat count per target: `{result.config.repeat_count}`",
            f"- First RTF <= 1.0: {_threshold_text(result.rows, 1.0)}",
            f"- First RTF <= 0.85: {_threshold_text(result.rows, 0.85)}",
            f"- First RTF <= 0.7: {_threshold_text(result.rows, 0.7)}",
            "",
            "| Target Chars | Actual Chars | Units | Repeats | Gen s | Playback s | RTF | Max Smooth Speed | Peak VRAM Reserved GB | Device VRAM Used After GB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.target_chars),
                    str(row.actual_chars),
                    str(row.unit_count),
                    str(row.repeat_count),
                    _fmt(row.generation_seconds),
                    _fmt(row.playback_seconds),
                    _fmt(row.rtf_at_1x),
                    _fmt(row.max_smooth_speed),
                    _fmt(row.peak_vram_reserved_gb),
                    _fmt(row.device_vram_used_after_gb),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _threshold_text(rows: Sequence[SweepRow], threshold: float) -> str:
    for row in rows:
        if row.rtf_at_1x is not None and row.rtf_at_1x <= threshold:
            return f"{row.target_chars} target chars, {row.actual_chars} actual chars"
    return "not reached"


def _avg(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def _max_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def _max_optional_int(values: Sequence[Optional[int]]) -> Optional[int]:
    numeric = [int(value) for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def _bytes_to_gb(value: int) -> Optional[float]:
    if value <= 0:
        return None
    return round(value / float(1024**3), 3)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"
