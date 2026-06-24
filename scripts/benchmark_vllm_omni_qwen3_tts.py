from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


LINUX_PATH = (
    "/opt/ebook-vllm-omni-venv/lib/python3.12/site-packages/nvidia/cu13/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
)
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class VoiceClonePromptItem:
    def __init__(
        self,
        ref_code: Any = None,
        ref_spk_embedding: Any = None,
        x_vector_only_mode: bool = True,
        icl_mode: bool = False,
        ref_text: str = "",
    ) -> None:
        self.ref_code = ref_code
        self.ref_spk_embedding = ref_spk_embedding
        self.x_vector_only_mode = x_vector_only_mode
        self.icl_mode = icl_mode
        self.ref_text = ref_text


@dataclass(frozen=True)
class Unit:
    unit_id: int
    text: str
    role: str
    role_id: str
    voice_config_path: str


@dataclass(frozen=True)
class Window:
    target_chars: int
    units: list[Unit]

    @property
    def actual_chars(self) -> int:
        return sum(len(unit.text) for unit in self.units)

    @property
    def word_count(self) -> int:
        return sum(len(unit.text.split()) for unit in self.units)

    @property
    def role_ids(self) -> list[str]:
        return sorted({unit.role_id for unit in self.units})

    @property
    def voice_config_paths(self) -> list[str]:
        return sorted({unit.voice_config_path for unit in self.units if unit.voice_config_path})


def main() -> int:
    _force_clean_linux_path()
    _install_qvp_pickle_shim()
    args = _parse_args()
    book_root = Path(args.book_root)
    logs_dir = book_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or f"vllm_omni_window_sweep_{args.chapter}"

    units = _load_units(book_root, args.chapter)
    windows = _build_windows(units, args.start_chars, args.step_chars, args.max_targets)
    voice_cache: dict[str, dict[str, Any]] = {}
    for window in windows:
        for unit in window.units:
            if unit.voice_config_path:
                _voice_prompt(book_root, unit.voice_config_path, voice_cache)

    from vllm_omni import Omni

    sampler = VramSampler(args.vram_poll_seconds)
    device_vram_after_init_gb: float | None = None
    started_init = time.perf_counter()
    omni: Any | None = None
    rows: list[dict[str, Any]] = []
    stop_reason = "max_targets" if args.max_targets is not None and len(windows) >= args.max_targets else "chapter_exhausted"
    try:
        omni = Omni(
            model=args.model,
            query_type="Base",
            output_dir=str(logs_dir / f"{output_stem}_tmp_audio"),
            log_stats=False,
            stage_configs_path=args.stage_configs_path,
            stage_init_timeout=args.stage_init_timeout,
            init_timeout=args.init_timeout,
            batch_timeout=args.batch_timeout,
            batch_size=1,
        )
        init_seconds = time.perf_counter() - started_init
        device_vram_after_init_gb = _gpu_used_gb()
        sampler.start()
        for window in windows:
            if args.warmup_text:
                warmup_unit = Unit(
                    unit_id=-1,
                    text=args.warmup_text,
                    role=window.units[0].role,
                    role_id=window.units[0].role_id,
                    voice_config_path=window.units[0].voice_config_path,
                )
                _generate_window(
                    omni=omni,
                    window=Window(target_chars=len(args.warmup_text), units=[warmup_unit]),
                    book_root=book_root,
                    voice_cache=voice_cache,
                    playback_speed=args.playback_speed,
                    model=args.model,
                )

            repeats = [
                _generate_window(
                    omni=omni,
                    window=window,
                    book_root=book_root,
                    voice_cache=voice_cache,
                    playback_speed=args.playback_speed,
                    model=args.model,
                )
                for _ in range(args.repeat_count)
            ]
            row = _average_rows(window, repeats)
            rows.append(row)
            if _row_peak_vram(row) >= args.max_vram_gb:
                stop_reason = "vram_limit"
                break
    finally:
        if omni is not None:
            shutdown = getattr(omni, "shutdown", None)
            if callable(shutdown):
                shutdown()
        sampler.stop()

    payload = {
        "backend": "vllm_omni_offline",
        "model": args.model,
        "init_seconds": init_seconds,
        "device_vram_after_init_gb": device_vram_after_init_gb,
        "device_vram_peak_gb": sampler.peak_gb,
        "config": {
            "chapter": args.chapter,
            "start_chars": args.start_chars,
            "step_chars": args.step_chars,
            "max_targets": args.max_targets,
            "repeat_count": args.repeat_count,
            "warmup_text": args.warmup_text,
            "playback_speed": args.playback_speed,
            "max_vram_gb": args.max_vram_gb,
            "stage_configs_path": args.stage_configs_path,
            "output_stem": output_stem,
            "vram_poll_seconds": args.vram_poll_seconds,
        },
        "rows": rows,
        "stop_reason": stop_reason,
    }
    json_path = logs_dir / f"{output_stem}.json"
    csv_path = logs_dir / f"{output_stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows), "stop_reason": stop_reason}))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Chapter read-along windows with vLLM-Omni Qwen3-TTS.")
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--start-chars", type=int, default=100)
    parser.add_argument("--step-chars", type=int, default=100)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--warmup-text", default="Test")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--max-vram-gb", type=float, default=10.0)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-stem")
    parser.add_argument("--vram-poll-seconds", type=float, default=0.25)
    parser.add_argument("--stage-configs-path")
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--batch-timeout", type=int, default=5)
    return parser.parse_args()


def _force_clean_linux_path() -> None:
    os.environ["PATH"] = LINUX_PATH
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _install_qvp_pickle_shim() -> None:
    qwen_mod = types.ModuleType("qwen_tts")
    inference_mod = types.ModuleType("qwen_tts.inference")
    model_mod = types.ModuleType("qwen_tts.inference.qwen3_tts_model")
    VoiceClonePromptItem.__module__ = "qwen_tts.inference.qwen3_tts_model"
    model_mod.VoiceClonePromptItem = VoiceClonePromptItem
    sys.modules.setdefault("qwen_tts", qwen_mod)
    sys.modules.setdefault("qwen_tts.inference", inference_mod)
    sys.modules.setdefault("qwen_tts.inference.qwen3_tts_model", model_mod)


def _load_units(book_root: Path, chapter: str) -> list[Unit]:
    payload = json.loads((book_root / "read_along" / f"{chapter}.units.json").read_text(encoding="utf-8"))
    return [
        Unit(
            unit_id=int(item["unit_id"]),
            text=str(item["text"]),
            role=str(item["role"]),
            role_id=str(item["role_id"]),
            voice_config_path=str(item.get("voice_config_path") or ""),
        )
        for item in payload["units"]
        if str(item.get("text") or "").strip()
    ]


def _build_windows(
    units: list[Unit],
    start_chars: int,
    step_chars: int,
    max_targets: int | None,
) -> list[Window]:
    windows: list[Window] = []
    target = int(start_chars)
    last_ids: tuple[int, ...] | None = None
    while units:
        selected: list[Unit] = []
        total = 0
        for unit in units:
            chars = len(unit.text)
            if selected and total + chars > target:
                break
            selected.append(unit)
            total += chars
            if total >= target:
                break
        ids = tuple(unit.unit_id for unit in selected)
        if ids != last_ids:
            windows.append(Window(target_chars=target, units=selected))
            last_ids = ids
            if max_targets is not None and len(windows) >= max_targets:
                break
            if len(selected) >= len(units):
                break
        target += max(1, int(step_chars))
    return windows


def _generate_window(
    *,
    omni: Any,
    window: Window,
    book_root: Path,
    voice_cache: dict[str, dict[str, Any]],
    playback_speed: float,
    model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    gpu_before = _gpu_used_gb()
    prompts = [_input_for_unit(book_root, unit, voice_cache, model) for unit in window.units]
    audio_seconds_by_unit: list[float | None] = [None] * len(window.units)
    sample_counts_by_unit: list[int | None] = [None] * len(window.units)
    completion_seconds_by_unit: list[float | None] = [None] * len(window.units)
    request_ids_by_unit: list[str | None] = [None] * len(window.units)
    for stage_output in omni.generate(prompts):
        completed_at = time.perf_counter() - started
        output = stage_output.request_output
        request_id = str(output.request_id)
        unit_index = _request_index(request_id)
        mm = output.outputs[0].multimodal_output
        seconds, samples = _audio_shape(mm)
        if unit_index is None or unit_index >= len(window.units):
            continue
        audio_seconds_by_unit[unit_index] = seconds
        sample_counts_by_unit[unit_index] = samples
        completion_seconds_by_unit[unit_index] = completed_at
        request_ids_by_unit[unit_index] = request_id
    elapsed = time.perf_counter() - started
    gpu_after = _gpu_used_gb()
    audio_seconds = [float(value or 0.0) for value in audio_seconds_by_unit]
    sample_counts = [int(value or 0) for value in sample_counts_by_unit]
    completion_seconds = [float(value or elapsed) for value in completion_seconds_by_unit]
    playback_seconds = sum(audio_seconds) / max(float(playback_speed), 0.001)
    rtf = elapsed / playback_seconds if playback_seconds > 0 else None
    unit_metrics = _unit_metrics(
        window=window,
        audio_seconds=audio_seconds,
        completion_seconds=completion_seconds,
        playback_speed=playback_speed,
        request_ids=request_ids_by_unit,
    )
    return {
        "generation_seconds": elapsed,
        "playback_seconds": playback_seconds,
        "rtf_at_1x": rtf,
        "max_smooth_speed": playback_seconds / elapsed if elapsed > 0 else None,
        "audio_seconds": audio_seconds,
        "sample_counts": sample_counts,
        "completion_seconds": completion_seconds,
        "request_ids": request_ids_by_unit,
        "unit_metrics": unit_metrics,
        "device_vram_used_before_gb": gpu_before,
        "device_vram_used_after_gb": gpu_after,
    }


def _request_index(request_id: str) -> int | None:
    prefix = request_id.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def _unit_metrics(
    *,
    window: Window,
    audio_seconds: list[float],
    completion_seconds: list[float],
    playback_speed: float,
    request_ids: list[str | None],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    cumulative_playback = 0.0
    effective_speed = max(float(playback_speed), 0.001)
    for idx, unit in enumerate(window.units):
        audio = float(audio_seconds[idx]) if idx < len(audio_seconds) else 0.0
        completion = float(completion_seconds[idx]) if idx < len(completion_seconds) else 0.0
        playback = audio / effective_speed
        cumulative_playback += playback
        metrics.append(
            {
                "unit_index": idx,
                "unit_id": unit.unit_id,
                "chars": len(unit.text),
                "words": len(unit.text.split()),
                "role_id": unit.role_id,
                "voice_config_path": unit.voice_config_path,
                "audio_seconds": audio,
                "completion_seconds": completion,
                "rtf_at_1x": completion / audio if audio > 0 else None,
                "max_smooth_speed": audio / completion if completion > 0 else None,
                "cumulative_playback_seconds": cumulative_playback,
                "readiness_margin_seconds": cumulative_playback - completion,
                "request_id": request_ids[idx] if idx < len(request_ids) else None,
            }
        )
    return metrics


def _input_for_unit(book_root: Path, unit: Unit, voice_cache: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    voice_prompt = _voice_prompt(book_root, unit.voice_config_path, voice_cache)
    info = {
        "task_type": ["Base"],
        "text": [unit.text],
        "language": ["English"],
        "speaker": [unit.role_id],
        "x_vector_only_mode": [True],
        "voice_clone_prompt": [voice_prompt],
        "max_new_tokens": [2048],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(info, model),
        "additional_information": info,
    }


def _voice_prompt(book_root: Path, voice_path: str, voice_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if voice_path in voice_cache:
        return voice_cache[voice_path]
    path = book_root / voice_path
    item = torch.load(path, map_location="cpu", weights_only=False)[0]
    embedding = item.ref_spk_embedding.detach().to(dtype=torch.float32).reshape(-1).tolist()
    prompt = {
        "ref_code": None,
        "ref_spk_embedding": embedding,
        "icl_mode": False,
        "ref_text": getattr(item, "ref_text", "") or "This is the reference voice for this character.",
    }
    voice_cache[voice_path] = prompt
    return prompt


def _estimate_prompt_len(info: dict[str, Any], model: str) -> int:
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    if not hasattr(_estimate_prompt_len, "_cache"):
        _estimate_prompt_len._cache = {}  # type: ignore[attr-defined]
    cache = _estimate_prompt_len._cache  # type: ignore[attr-defined]
    if model not in cache:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left")
        cfg = Qwen3TTSConfig.from_pretrained(model, trust_remote_code=True)
        cache[model] = (tok, getattr(cfg, "talker_config", None))
    tok, tcfg = cache[model]
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=info,
        task_type="Base",
        tokenize_prompt=lambda text: tok(text, padding=False)["input_ids"],
        codec_language_id=getattr(tcfg, "codec_language_id", None),
        spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
        estimate_ref_code_len=None,
    )


def _audio_shape(mm: dict[str, Any]) -> tuple[float, int]:
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    samples = int(audio_tensor.numel())
    return samples / sr, samples


def _gpu_used_gb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    first = out.strip().splitlines()[0].strip()
    return round(float(first) / 1024.0, 3)


class VramSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.05, float(interval_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._peak_gb: float | None = None

    @property
    def peak_gb(self) -> float | None:
        with self._lock:
            return self._peak_gb

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _gpu_used_gb()
            if value is not None:
                with self._lock:
                    self._peak_gb = value if self._peak_gb is None else max(self._peak_gb, value)
            self._stop.wait(self.interval_seconds)


def _average_rows(window: Window, rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = rows
    generation = sum(float(row["generation_seconds"]) for row in success_rows) / len(success_rows)
    playback = sum(float(row["playback_seconds"]) for row in success_rows) / len(success_rows)
    audio_seconds = [
        sum(float(row["audio_seconds"][idx]) for row in success_rows) / len(success_rows)
        for idx in range(len(success_rows[0]["audio_seconds"]))
    ]
    completion_seconds = [
        sum(float(row["completion_seconds"][idx]) for row in success_rows) / len(success_rows)
        for idx in range(len(success_rows[0]["completion_seconds"]))
    ]
    gpu_before = _avg_optional(row["device_vram_used_before_gb"] for row in success_rows)
    gpu_after = _avg_optional(row["device_vram_used_after_gb"] for row in success_rows)
    unit_metrics = _average_unit_metrics(window, success_rows)
    return {
        "target_chars": window.target_chars,
        "actual_chars": window.actual_chars,
        "word_count": window.word_count,
        "unit_count": len(window.units),
        "unit_ids": [unit.unit_id for unit in window.units],
        "role_ids": window.role_ids,
        "voice_config_paths": window.voice_config_paths,
        "generation_seconds": generation,
        "playback_seconds": playback,
        "rtf_at_1x": generation / playback if playback > 0 else None,
        "max_smooth_speed": playback / generation if generation > 0 else None,
        "device_vram_used_before_gb": gpu_before,
        "device_vram_used_after_gb": gpu_after,
        "audio_seconds": audio_seconds,
        "completion_seconds": completion_seconds,
        "unit_metrics": unit_metrics,
        "repeat_count": len(success_rows),
        "success": True,
    }


def _average_unit_metrics(window: Window, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    averaged: list[dict[str, Any]] = []
    for idx, unit in enumerate(window.units):
        unit_rows = [row["unit_metrics"][idx] for row in rows]
        audio = sum(float(item["audio_seconds"]) for item in unit_rows) / len(unit_rows)
        completion = sum(float(item["completion_seconds"]) for item in unit_rows) / len(unit_rows)
        cumulative = sum(float(item["cumulative_playback_seconds"]) for item in unit_rows) / len(unit_rows)
        averaged.append(
            {
                "unit_index": idx,
                "unit_id": unit.unit_id,
                "chars": len(unit.text),
                "words": len(unit.text.split()),
                "role_id": unit.role_id,
                "voice_config_path": unit.voice_config_path,
                "audio_seconds": audio,
                "completion_seconds": completion,
                "rtf_at_1x": completion / audio if audio > 0 else None,
                "max_smooth_speed": audio / completion if completion > 0 else None,
                "cumulative_playback_seconds": cumulative,
                "readiness_margin_seconds": cumulative - completion,
            }
        )
    return averaged


def _avg_optional(values: Any) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 3)


def _row_peak_vram(row: dict[str, Any]) -> float:
    return max(float(row.get("device_vram_used_before_gb") or 0), float(row.get("device_vram_used_after_gb") or 0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
        "device_vram_used_before_gb",
        "device_vram_used_after_gb",
        "audio_seconds",
        "completion_seconds",
        "unit_metrics",
        "repeat_count",
        "success",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row.get(field), list) else row.get(field) for field in fields})


if __name__ == "__main__":
    raise SystemExit(main())
