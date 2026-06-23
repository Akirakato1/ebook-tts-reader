from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.text_normalization import normalize_tts_text


HF_MODEL_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


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
        candidates = [
            self.model_root / folder_name,
            self.model_root / "Qwen" / folder_name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return repo_id

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
        self._voice_prompt_cache: Dict[str, Any] = {}

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        self.role_voice_paths.setdefault(role_id, voice_path)
        if voice_path.exists() and not voice_record.get("_force_regenerate"):
            return voice_path
        seed = int(voice_record.get("voice_identity", {}).get("seed", 0))
        instruct = str(voice_record["voice_profile"]["qwen_instruct"])
        text = "This is the reference voice for this character."

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
        generated = self._generate_blocks(blocks)
        try:
            yield generated
        finally:
            del generated
            gc.collect()
            self._clear_device_cache()

    def _build_generation_blocks(self, jobs: List[Dict]) -> List[_GenerationBlock]:
        blocks: List[_GenerationBlock] = []
        current_jobs: List[Dict] = []
        current_role: Optional[str] = None
        current_voice_path: Optional[Path] = None

        for job in jobs:
            role = str(job["role"])
            voice_path = self.role_voice_paths[role]
            if current_jobs and (
                role != current_role
                or voice_path != current_voice_path
            ):
                blocks.append(
                    _GenerationBlock(
                        role=current_role or "",
                        voice_path=current_voice_path or voice_path,
                        jobs=current_jobs,
                    )
                )
                current_jobs = []
            current_role = role
            current_voice_path = voice_path
            current_jobs.append(job)

        if current_jobs:
            blocks.append(
                _GenerationBlock(
                    role=current_role or "",
                    voice_path=current_voice_path or self.role_voice_paths[str(current_jobs[0]["role"])],
                    jobs=current_jobs,
                )
            )
        return blocks

    def _generate_blocks(self, blocks: List[_GenerationBlock]) -> List[GeneratedSentenceAudio]:
        prompts = [self._load_voice_prompt(block.voice_path) for block in blocks]
        wavs, sample_rate = self.model.generate_voice_clone(
            text=[block.text for block in blocks],
            language=[self.language] * len(blocks),
            voice_clone_prompt=prompts,
        )
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
