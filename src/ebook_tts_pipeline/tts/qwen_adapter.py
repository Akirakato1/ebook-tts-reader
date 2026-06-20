from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio


class QwenTtsAdapter:
    def __init__(
        self,
        model: Optional[Any] = None,
        torch_module: Optional[Any] = None,
        role_voice_paths: Optional[Dict[str, Path]] = None,
        language: str = "auto",
    ) -> None:
        self.model = model if model is not None else self._load_default_model()
        self.torch = torch_module if torch_module is not None else self._load_torch()
        self.role_voice_paths = role_voice_paths or {}
        self.language = language

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        self.role_voice_paths.setdefault(role_id, voice_path)
        if voice_path.exists():
            return voice_path
        seed = int(voice_record.get("voice_identity", {}).get("seed", 0))
        instruct = str(voice_record["voice_profile"]["qwen_instruct"])
        text = "This is the reference voice for this character."

        self._set_seed(seed)
        wavs, sample_rate = self.model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=self.language,
        )
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
        current_batch: List[Dict] = []
        current_role: Optional[str] = None

        for job in jobs:
            role = str(job["role"])
            if current_batch and role != current_role:
                generated.extend(self._generate_role_batch(current_role or "", current_batch))
                current_batch = []
            current_role = role
            current_batch.append(job)

        if current_batch:
            generated.extend(self._generate_role_batch(current_role or "", current_batch))

        return generated

    def _generate_role_batch(self, role: str, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
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
            )
            for index, job in enumerate(jobs)
        ]

    def _set_seed(self, seed: int) -> None:
        if hasattr(self.torch, "manual_seed"):
            self.torch.manual_seed(seed)

    def _load_torch(self) -> Any:
        import torch

        return torch

    def _load_default_model(self) -> Any:
        raise RuntimeError(
            "Qwen model loading must be wired with local qwen_tts runtime paths before real TTS use."
        )
