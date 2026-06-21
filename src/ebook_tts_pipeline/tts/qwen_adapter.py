from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


HF_MODEL_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


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
        max_batch_size: int = 8,
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
        self.max_batch_size = max(1, int(max_batch_size))

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
        return voice_path

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        generated: List[GeneratedSentenceAudio] = []
        for batch in self.generate_sentence_batches(jobs):
            generated.extend(batch)
        return generated

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        current_batch: List[Dict] = []
        current_role: Optional[str] = None

        for job in jobs:
            role = str(job["role"])
            if current_batch and role != current_role:
                yield from self._generate_role_batches(current_role or "", current_batch)
                current_batch = []
            current_role = role
            current_batch.append(job)

        if current_batch:
            yield from self._generate_role_batches(current_role or "", current_batch)

    def _generate_role_batches(self, role: str, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        for start in range(0, len(jobs), self.max_batch_size):
            generated = self._generate_role_chunk(role, jobs[start:start + self.max_batch_size])
            try:
                yield generated
            finally:
                del generated
                gc.collect()
                self._clear_device_cache()

    def _generate_role_chunk(self, role: str, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        voice_path = self.role_voice_paths[role]
        prompt = self.torch.load(voice_path, map_location="cpu", weights_only=False)
        wavs, sample_rate = self.model.generate_voice_clone(
            text=[str(job["text"]) for job in jobs],
            language=[self.language] * len(jobs),
            voice_clone_prompt=prompt,
        )
        return [
            GeneratedSentenceAudio(
                sentence_idx=int(job["sentence_idx"]),
                role=role,
                speech_type=str(job["type"]),
                samples=np.asarray(wavs[index], dtype=np.float32),
                sample_rate=sample_rate,
                unit_idx=int(job.get("unit_idx", job["sentence_idx"])),
            )
            for index, job in enumerate(jobs)
        ]

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
