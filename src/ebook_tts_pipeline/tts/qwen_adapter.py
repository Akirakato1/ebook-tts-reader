from __future__ import annotations

import gc
import importlib.util
import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.text_normalization import normalize_tts_text


HF_MODEL_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
MAX_GENERATION_BLOCK_CHARS = 0
MAX_GENERATION_BLOCKS_PER_CALL = 0
CACHE_CLEAR_INTERVAL = 8


@dataclass(frozen=True)
class _GenerationBlock:
    role: str
    voice_path: Path
    jobs: List[Dict]

    @property
    def text(self) -> str:
        return " ".join(_normalize_block_text(str(job["text"])) for job in self.jobs if str(job["text"]).strip())


class QwenTtsRuntime:
    def __init__(
        self,
        qwen_model_cls: Optional[Any] = None,
        torch_module: Optional[Any] = None,
        model_root: Optional[Path] = None,
        model_choice: str = "1.7B",
        device: str = "auto",
        precision: str = "bf16",
        attention: str = "auto",
    ) -> None:
        self.qwen_model_cls = qwen_model_cls
        self.torch = torch_module
        self.model_root = Path(model_root) if model_root else Path("models") / "qwen-tts"
        self.model_choice = model_choice
        self.device = device
        self.precision = precision
        self.attention = attention
        self._models: Dict[str, Any] = {}

    def generate_voice_design(self, *args, **kwargs):
        return self._model("VoiceDesign").generate_voice_design(*args, **kwargs)

    def create_voice_clone_prompt(self, *args, **kwargs):
        return self._model("Base").create_voice_clone_prompt(*args, **kwargs)

    def generate_voice_clone(self, *args, **kwargs):
        return self._model("Base").generate_voice_clone(*args, **kwargs)

    def unload_model(self, model_type: str) -> None:
        if model_type in self._models:
            del self._models[model_type]
            gc.collect()
            self._clear_device_cache()

    def _model(self, model_type: str) -> Any:
        if model_type not in self._models:
            self._models[model_type] = self._load_model(model_type)
        return self._models[model_type]

    def _load_model(self, model_type: str) -> Any:
        qwen_model_cls = self.qwen_model_cls or self._load_qwen_model_cls()
        torch_module = self.torch or self._load_torch()
        source = self._resolve_model_source(model_type)
        kwargs = {
            "device_map": self._device_map(torch_module),
            "dtype": self._dtype(torch_module),
        }
        attention = self._attention_param()
        if attention:
            kwargs["attn_implementation"] = attention
        return qwen_model_cls.from_pretrained(str(source), **kwargs)

    def _resolve_model_source(self, model_type: str) -> str:
        repo_id = HF_MODEL_MAP.get((model_type, self.model_choice))
        if repo_id is None:
            raise RuntimeError(f"Unsupported Qwen3-TTS model selection: {model_type} {self.model_choice}")
        folder_name = repo_id.split("/")[-1]
        candidates = self._local_model_candidates(folder_name)
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                log_runtime_step(
                    "qwen_model_resolved",
                    model_type=model_type,
                    model_choice=self.model_choice,
                    source=candidate,
                )
                return str(candidate)
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            f"Local Qwen3-TTS model not found for {model_type} {self.model_choice}. "
            f"Checked: {checked}. Refusing to download {repo_id}; set EBOOK_TTS_QWEN_MODEL_ROOT "
            "to the local models/qwen-tts folder."
        )

    def _local_model_candidates(self, folder_name: str) -> List[Path]:
        roots = [self.model_root]
        if not self.model_root.is_absolute():
            roots.append(Path(__file__).resolve().parents[3] / self.model_root)
        candidates: List[Path] = []
        seen = set()
        for root in roots:
            for candidate in (root / folder_name, root / "Qwen" / folder_name):
                key = str(candidate)
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
        return candidates

    def _device_map(self, torch_module: Any) -> Any:
        device = self._resolved_device(torch_module)
        if device == "cuda":
            return "cuda"
        if device == "xpu":
            return {"": "xpu:0"}
        if device == "mps":
            return "mps"
        return "cpu"

    def _dtype(self, torch_module: Any) -> Any:
        if self.precision == "bf16":
            return torch_module.bfloat16
        return torch_module.float32

    def _attention_param(self) -> Optional[str]:
        if self.attention == "flash_attn":
            return "flash_attention_2"
        if self.attention in {"sdpa", "eager"}:
            return self.attention
        if self.attention == "auto":
            if self.precision == "bf16" and importlib.util.find_spec("flash_attn") is not None:
                return "flash_attention_2"
            return "sdpa"
        return None

    def _resolved_device(self, torch_module: Any) -> str:
        if self.device != "auto":
            return self.device
        if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
            return "xpu"
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_qwen_model_cls(self) -> Any:
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel

    def _load_torch(self) -> Any:
        import torch

        return torch

    def _clear_device_cache(self) -> None:
        torch_module = self.torch or self._load_torch()
        if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "empty_cache"):
            torch_module.cuda.empty_cache()
        if hasattr(torch_module, "xpu") and hasattr(torch_module.xpu, "empty_cache"):
            torch_module.xpu.empty_cache()


class QwenTtsAdapter:
    def __init__(
        self,
        model: Optional[Any] = None,
        torch_module: Optional[Any] = None,
        role_voice_paths: Optional[Dict[str, Path]] = None,
        language: str = "auto",
        model_root: Optional[str] = None,
        model_choice: str = "1.7B",
        device: str = "auto",
        precision: str = "bf16",
        attention: str = "auto",
        max_new_tokens: int = 2048,
        max_generation_block_chars: int = MAX_GENERATION_BLOCK_CHARS,
        max_generation_blocks_per_call: int = MAX_GENERATION_BLOCKS_PER_CALL,
        cache_clear_interval: int = CACHE_CLEAR_INTERVAL,
        streaming_text_mode: bool = True,
        performance_log_path: Optional[Path] = None,
        adaptive_memory_target_bytes: Optional[int] = None,
    ) -> None:
        self.torch = torch_module if torch_module is not None else self._load_torch()
        self.model = model if model is not None else self._load_default_model(
            model_root=model_root,
            model_choice=model_choice,
            device=device,
            precision=precision,
            attention=attention,
        )
        self.role_voice_paths = role_voice_paths or {}
        self.language = language
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.max_generation_block_chars = max(0, int(max_generation_block_chars))
        self.max_generation_blocks_per_call = max(0, int(max_generation_blocks_per_call))
        self.cache_clear_interval = max(0, int(cache_clear_interval))
        self.streaming_text_mode = bool(streaming_text_mode)
        self.performance_log_path = Path(performance_log_path) if performance_log_path else None
        self.adaptive_memory_target_bytes = (
            max(1, int(adaptive_memory_target_bytes))
            if adaptive_memory_target_bytes is not None
            else None
        )
        self._voice_prompt_cache: Dict[str, Any] = {}
        self._generated_batch_count = 0

    def ensure_voice(
        self,
        role_id: str,
        voice_record: Dict,
        voice_path: Path,
        sample_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
    ) -> Path:
        self.role_voice_paths.setdefault(role_id, voice_path)
        sample_path = Path(sample_path) if sample_path is not None else None
        sample_missing = sample_path is not None and not sample_path.exists()
        if voice_path.exists() and not voice_record.get("_force_regenerate") and not sample_missing:
            return voice_path
        seed = int(voice_record.get("voice_identity", {}).get("seed", 0))
        instruct = str(voice_record["voice_profile"]["qwen_instruct"])
        text = str(reference_text or "This is the reference voice for this character.")

        if hasattr(self.model, "unload_model"):
            self.model.unload_model("Base")
        self._set_seed(seed)
        try:
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=self.language,
            )
        finally:
            if hasattr(self.model, "unload_model"):
                self.model.unload_model("VoiceDesign")
        if sample_path is not None:
            _write_wav_file(sample_path, np.asarray(wavs[0], dtype=np.float32), int(sample_rate))
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=(wavs[0], sample_rate),
            ref_text=text,
            x_vector_only_mode=True,
        )
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(prompt, voice_path)
        self._voice_prompt_cache.pop(str(voice_path), None)
        return voice_path

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        generated: List[GeneratedSentenceAudio] = []
        for batch in self.generate_sentence_batches(jobs):
            generated.extend(batch)
        return generated

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        blocks = self._build_generation_blocks(jobs)
        for block_batch in self._generation_block_batches(blocks):
            generated = self._generate_blocks(block_batch)
            try:
                yield generated
            finally:
                del generated
                gc.collect()
                self._clear_device_cache_if_due()

    def _build_generation_blocks(self, jobs: List[Dict]) -> List[_GenerationBlock]:
        blocks: List[_GenerationBlock] = []
        current_jobs: List[Dict] = []
        current_role: Optional[str] = None
        current_voice_path: Optional[Path] = None
        current_chars = 0

        for job in jobs:
            role = str(job["role"])
            voice_path = self._voice_path_for_job(job)
            job_chars = _audio_timeline_weight(str(job["text"]))
            separator_chars = 1 if current_jobs else 0
            would_exceed_block = self.max_generation_block_chars > 0 and current_jobs and (
                current_chars + separator_chars + job_chars > self.max_generation_block_chars
            )
            if current_jobs and (
                role != current_role
                or voice_path != current_voice_path
                or would_exceed_block
            ):
                blocks.append(
                    _GenerationBlock(
                        role=current_role or "",
                        voice_path=current_voice_path or voice_path,
                        jobs=current_jobs,
                    )
                )
                current_jobs = []
                current_chars = 0
            current_role = role
            current_voice_path = voice_path
            separator_chars = 1 if current_jobs else 0
            current_chars += separator_chars + job_chars
            current_jobs.append(job)

        if current_jobs:
            blocks.append(
                _GenerationBlock(
                    role=current_role or "",
                    voice_path=current_voice_path or self._voice_path_for_job(current_jobs[0]),
                    jobs=current_jobs,
                )
            )
        return blocks

    def _voice_path_for_job(self, job: Dict) -> Path:
        for key_name in ("role_id", "role"):
            key = str(job.get(key_name, "")).strip()
            if key and key in self.role_voice_paths:
                return self.role_voice_paths[key]
        role = str(job.get("role", ""))
        role_id = str(job.get("role_id", ""))
        raise KeyError(f"No voice path registered for role={role!r}, role_id={role_id!r}")

    def _generation_block_batches(self, blocks: List[_GenerationBlock]) -> Iterator[List[_GenerationBlock]]:
        if self.max_generation_blocks_per_call <= 0:
            if blocks:
                yield blocks
            return

        current: List[_GenerationBlock] = []
        for block in blocks:
            if current and (
                len(current) >= self.max_generation_blocks_per_call
            ):
                yield current
                current = []
            current.append(block)
        if current:
            yield current

    def _generate_blocks(self, blocks: List[_GenerationBlock]) -> List[GeneratedSentenceAudio]:
        texts = [block.text for block in blocks]
        prompts = [self._load_voice_prompt(block.voice_path) for block in blocks]
        event = self._start_performance_event(blocks, texts)
        try:
            wavs, sample_rate = self.model.generate_voice_clone(
                text=texts,
                language=[self.language] * len(blocks),
                voice_clone_prompt=prompts,
                max_new_tokens=self.max_new_tokens,
                non_streaming_mode=not self.streaming_text_mode,
            )
            self._finish_performance_event(event, wavs, sample_rate=sample_rate)
        except Exception as exc:
            self._finish_performance_event(event, None, error=str(exc), sample_rate=None)
            raise
        generated: List[GeneratedSentenceAudio] = []
        for index, block in enumerate(blocks):
            generated.extend(
                self._split_block_audio(
                    block=block,
                    samples=np.asarray(wavs[index], dtype=np.float32),
                    sample_rate=sample_rate,
                )
            )
        return generated

    def _start_performance_event(self, blocks: List[_GenerationBlock], texts: List[str]) -> Dict[str, Any]:
        jobs = [job for block in blocks for job in block.jobs]
        section_job_counts = _section_job_counts(jobs)
        event = {
            "timestamp": time.time(),
            "batch_size": len(blocks),
            "generation_block_count": len(blocks),
            "max_new_tokens": self.max_new_tokens,
            "roles": [block.role for block in blocks],
            "voice_config_paths": [str(block.voice_path) for block in blocks],
            "unique_voice_count": len({str(block.voice_path) for block in blocks}),
            "job_counts": [len(block.jobs) for block in blocks],
            "job_text_chars": [_audio_timeline_weight(str(job["text"])) for job in jobs],
            "max_job_chars": max((_audio_timeline_weight(str(job["text"])) for job in jobs), default=0),
            "role_switch_count": _role_switch_count(jobs),
            "text_chars": [len(text) for text in texts],
            "text_char_sum": sum(len(text) for text in texts),
            "text_char_max": max((len(text) for text in texts), default=0),
            "block_sentence_indices": [
                [int(job["sentence_idx"]) for job in block.jobs]
                for block in blocks
            ],
            "tts_section_indices": _section_indices(jobs),
            "tts_section_char_count_max": max(_section_char_counts(jobs), default=0),
            "tts_section_job_count_sum": sum(section_job_counts.values()),
            "_started_at": time.perf_counter(),
        }
        if self.performance_log_path is not None or self.adaptive_memory_target_bytes is not None:
            self._reset_cuda_peak_memory()
            event["cuda_before"] = self._cuda_memory_snapshot()
        return event

    def _finish_performance_event(
        self,
        event: Dict[str, Any],
        wavs: Optional[List[Any]],
        sample_rate: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        event["elapsed_seconds"] = time.perf_counter() - float(event.pop("_started_at"))
        event["success"] = error is None
        if error is not None:
            event["error"] = error
        if sample_rate is not None:
            event["sample_rate"] = sample_rate
        if wavs is not None:
            sample_counts = [int(len(wav)) for wav in wavs]
            event["sample_counts"] = sample_counts
            if sample_rate:
                event["audio_seconds"] = [count / sample_rate for count in sample_counts]
        if self.performance_log_path is not None or self.adaptive_memory_target_bytes is not None:
            event["cuda_after"] = self._cuda_memory_snapshot()
            self._adapt_generation_block_limit(event)
        if self.performance_log_path is not None:
            self._append_performance_event(event)

    def _append_performance_event(self, event: Dict[str, Any]) -> None:
        if self.performance_log_path is None:
            return
        self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.performance_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _adapt_generation_block_limit(self, event: Dict[str, Any]) -> None:
        if self.adaptive_memory_target_bytes is None:
            return
        cuda_after = event.get("cuda_after")
        if not isinstance(cuda_after, dict) or not cuda_after.get("available"):
            return

        target = self.adaptive_memory_target_bytes
        peak = max(
            int(cuda_after.get("max_memory_reserved", 0) or 0),
            int(cuda_after.get("max_memory_allocated", 0) or 0),
            int(cuda_after.get("memory_reserved", 0) or 0),
            int(cuda_after.get("memory_allocated", 0) or 0),
        )
        if peak <= 0:
            return

        previous_limit = self.max_generation_blocks_per_call
        observed_blocks = max(1, int(event.get("generation_block_count", 1)))
        next_limit = previous_limit
        reason = "unchanged"

        if peak > target:
            next_limit = max(1, int(observed_blocks * target / peak * 0.85))
            reason = "over_target"
        elif previous_limit > 0 and peak < int(target * 0.65):
            expanded = previous_limit + max(1, previous_limit // 2)
            next_limit = 0 if expanded >= 24 else expanded
            reason = "under_target_expand"

        self.max_generation_blocks_per_call = next_limit
        event["adaptive_memory_target_bytes"] = target
        event["adaptive_peak_memory_bytes"] = peak
        event["adaptive_previous_blocks_per_call"] = previous_limit
        event["adaptive_next_blocks_per_call"] = next_limit
        event["adaptive_reason"] = reason

    def _reset_cuda_peak_memory(self) -> None:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not _call_bool(getattr(cuda, "is_available", None)):
            return
        _call_if_available(cuda, "synchronize")
        _call_if_available(cuda, "reset_peak_memory_stats")

    def _cuda_memory_snapshot(self) -> Dict[str, Any]:
        cuda = getattr(self.torch, "cuda", None)
        if cuda is None or not _call_bool(getattr(cuda, "is_available", None)):
            return {"available": False}
        _call_if_available(cuda, "synchronize")
        snapshot: Dict[str, Any] = {"available": True}
        for name in (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            value = _call_if_available(cuda, name)
            if value is not None:
                snapshot[name] = int(value)
        mem_get_info = getattr(cuda, "mem_get_info", None)
        if callable(mem_get_info):
            free_bytes, total_bytes = mem_get_info()
            snapshot["mem_free"] = int(free_bytes)
            snapshot["mem_total"] = int(total_bytes)
        return snapshot

    def _load_voice_prompt(self, voice_path: Path) -> Any:
        cache_key = str(voice_path)
        if cache_key not in self._voice_prompt_cache:
            self._voice_prompt_cache[cache_key] = self._normalize_prompt_for_batch(
                self.torch.load(voice_path, map_location="cpu", weights_only=False)
            )
        return self._voice_prompt_cache[cache_key]

    def _normalize_prompt_for_batch(self, prompt: Any) -> Any:
        if isinstance(prompt, dict) and "prompt" in prompt:
            prompt = prompt["prompt"]
        if isinstance(prompt, list) and len(prompt) == 1:
            return prompt[0]
        return prompt

    def _split_block_audio(
        self,
        block: _GenerationBlock,
        samples: np.ndarray,
        sample_rate: int,
    ) -> List[GeneratedSentenceAudio]:
        if len(block.jobs) == 1:
            job = block.jobs[0]
            return [
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    role=block.role,
                    speech_type=str(job["type"]),
                    samples=samples,
                    sample_rate=sample_rate,
                    unit_idx=int(job.get("unit_idx", job["sentence_idx"])),
                    voice_config_path=str(block.voice_path),
                )
            ]

        weights = [_audio_timeline_weight(str(job["text"])) for job in block.jobs]
        total_weight = sum(weights) or len(weights)
        boundaries = [0]
        consumed_weight = 0
        for weight in weights[:-1]:
            consumed_weight += weight
            boundaries.append(int(round(len(samples) * consumed_weight / total_weight)))
        boundaries.append(len(samples))

        generated: List[GeneratedSentenceAudio] = []
        for index, job in enumerate(block.jobs):
            start = boundaries[index]
            end = boundaries[index + 1]
            generated.append(
                GeneratedSentenceAudio(
                    sentence_idx=int(job["sentence_idx"]),
                    role=block.role,
                    speech_type=str(job["type"]),
                    samples=samples[start:end],
                    sample_rate=sample_rate,
                    unit_idx=int(job.get("unit_idx", job["sentence_idx"])),
                    pause_after_ms=0 if index < len(block.jobs) - 1 else None,
                    voice_config_path=str(block.voice_path),
                )
            )
        return generated

    def _set_seed(self, seed: int) -> None:
        if hasattr(self.torch, "manual_seed"):
            self.torch.manual_seed(seed)

    def _load_torch(self) -> Any:
        import torch

        return torch

    def _clear_device_cache(self) -> None:
        if hasattr(self.torch, "cuda") and hasattr(self.torch.cuda, "empty_cache"):
            self.torch.cuda.empty_cache()
        if hasattr(self.torch, "xpu") and hasattr(self.torch.xpu, "empty_cache"):
            self.torch.xpu.empty_cache()

    def _clear_device_cache_if_due(self) -> None:
        self._generated_batch_count += 1
        if self.cache_clear_interval <= 0:
            return
        if self._generated_batch_count % self.cache_clear_interval == 0:
            self._clear_device_cache()

    def _load_default_model(
        self,
        model_root: Optional[str],
        model_choice: str,
        device: str,
        precision: str,
        attention: str,
    ) -> Any:
        return QwenTtsRuntime(
            torch_module=self.torch,
            model_root=Path(model_root) if model_root else None,
            model_choice=model_choice,
            device=device,
            precision=precision,
            attention=attention,
        )


def _normalize_block_text(text: str) -> str:
    return normalize_tts_text(text)


def _audio_timeline_weight(text: str) -> int:
    return max(1, len(_normalize_block_text(text)))


def _section_indices(jobs: List[Dict]) -> List[int]:
    indices = {
        int(job["_tts_section_idx"])
        for job in jobs
        if "_tts_section_idx" in job and str(job.get("_tts_section_idx", "")).strip()
    }
    return sorted(indices)


def _section_char_counts(jobs: List[Dict]) -> List[int]:
    return [
        int(job["_tts_section_char_count"])
        for job in jobs
        if "_tts_section_char_count" in job and str(job.get("_tts_section_char_count", "")).strip()
    ]


def _section_job_counts(jobs: List[Dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for job in jobs:
        if "_tts_section_idx" not in job or "_tts_section_job_count" not in job:
            continue
        section_idx = int(job["_tts_section_idx"])
        counts[section_idx] = max(counts.get(section_idx, 0), int(job["_tts_section_job_count"]))
    return counts


def _role_switch_count(jobs: List[Dict]) -> int:
    switches = 0
    previous: Optional[str] = None
    for job in jobs:
        role = str(job.get("role_id") or job.get("role") or "")
        if previous is not None and role != previous:
            switches += 1
        previous = role
    return switches


def _write_wav_file(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples.astype(np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def _call_if_available(owner: Any, name: str) -> Any:
    func = getattr(owner, name, None)
    if callable(func):
        return func()
    return None


def _call_bool(func: Any) -> bool:
    if callable(func):
        return bool(func())
    return False
