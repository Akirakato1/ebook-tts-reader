from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_worker import encode_audio_item


LINUX_PATH = (
    "/opt/ebook-vllm-omni-venv/lib/python3.12/site-packages/nvidia/cu13/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
)


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


class VllmOmniWorkerState:
    def __init__(self) -> None:
        self.omni: Any = None
        self.model = ""
        self.book_root = Path(".")
        self.voice_cache: Dict[str, Dict[str, Any]] = {}

    def init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        _force_clean_linux_path()
        _install_qvp_pickle_shim()
        from vllm_omni import Omni

        self.book_root = Path(str(payload["book_root"]))
        self.model = str(payload["model"])
        self.omni = Omni(
            model=self.model,
            query_type="Base",
            output_dir=str(payload["output_dir"]),
            log_stats=False,
            stage_configs_path=str(payload["stage_configs_path"]),
            batch_size=1,
        )
        return {
            "initialized": True,
            "model": self.model,
            "stage_configs_path": str(payload["stage_configs_path"]),
        }

    def generate_sentences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        omni = self._require_omni()
        jobs = [dict(job) for job in payload["jobs"]]
        prompts = [self._input_for_job(job) for job in jobs]
        generated: List[Optional[GeneratedSentenceAudio]] = [None] * len(jobs)
        for stage_output in omni.generate(prompts):
            output = stage_output.request_output
            unit_index = _request_index(str(output.request_id))
            if unit_index is None or unit_index >= len(jobs):
                continue
            samples, sample_rate = _audio_samples(output.outputs[0].multimodal_output)
            job = jobs[unit_index]
            generated[unit_index] = GeneratedSentenceAudio(
                sentence_idx=int(job.get("sentence_idx", unit_index)),
                unit_idx=int(job["unit_idx"]) if job.get("unit_idx") is not None else unit_index,
                role=str(job.get("role", "")),
                speech_type=str(job.get("type", job.get("speech_type", "narration"))),
                samples=samples,
                sample_rate=sample_rate,
                pause_after_ms=job.get("pause_after_ms"),
                voice_config_path=str(job.get("voice_config_path") or ""),
            )
        missing = [idx for idx, item in enumerate(generated) if item is None]
        if missing:
            raise RuntimeError(f"vLLM-Omni did not return audio for job indices: {missing}")
        return {"items": [encode_audio_item(item) for item in generated if item is not None]}

    def shutdown(self) -> Dict[str, Any]:
        if self.omni is not None:
            shutdown = getattr(self.omni, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.omni = None
        self.voice_cache.clear()
        return {"shutdown": True}

    def _require_omni(self) -> Any:
        if self.omni is None:
            raise RuntimeError("vLLM-Omni worker has not been initialized.")
        return self.omni

    def _input_for_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        voice_path = str(job.get("voice_config_path") or "")
        if not voice_path:
            raise RuntimeError(f"Missing voice_config_path for role {job.get('role_id') or job.get('role')}.")
        info = {
            "task_type": ["Base"],
            "text": [str(job.get("text", ""))],
            "language": ["English"],
            "speaker": [str(job.get("role_id") or job.get("role") or "speaker")],
            "x_vector_only_mode": [True],
            "voice_clone_prompt": [self._voice_prompt(voice_path)],
            "max_new_tokens": [int(job.get("max_new_tokens", 2048))],
        }
        return {
            "prompt_token_ids": [0] * _estimate_prompt_len(info, self.model),
            "additional_information": info,
        }

    def _voice_prompt(self, voice_path: str) -> Dict[str, Any]:
        if voice_path in self.voice_cache:
            return self.voice_cache[voice_path]
        import torch

        item = torch.load(Path(voice_path), map_location="cpu", weights_only=False)[0]
        embedding = item.ref_spk_embedding.detach().to(dtype=torch.float32).reshape(-1).tolist()
        prompt = {
            "ref_code": None,
            "ref_spk_embedding": embedding,
            "icl_mode": False,
            "ref_text": getattr(item, "ref_text", "") or "This is the reference voice for this character.",
        }
        self.voice_cache[voice_path] = prompt
        return prompt


def handle_command(state: VllmOmniWorkerState, command: Dict[str, Any]) -> Dict[str, Any]:
    name = str(command.get("command", ""))
    payload = dict(command.get("payload", {}))
    if name == "init":
        return state.init(payload)
    if name == "generate_sentences":
        return state.generate_sentences(payload)
    if name == "shutdown":
        return state.shutdown()
    raise ValueError(f"Unknown worker command: {name}")


def _force_clean_linux_path() -> None:
    if os.name != "nt":
        os.environ["PATH"] = LINUX_PATH
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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


def _estimate_prompt_len(info: Dict[str, Any], model: str) -> int:
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    if not hasattr(_estimate_prompt_len, "_cache"):
        _estimate_prompt_len._cache = {}  # type: ignore[attr-defined]
    cache = _estimate_prompt_len._cache  # type: ignore[attr-defined]
    if model not in cache:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side="left")
        config = Qwen3TTSConfig.from_pretrained(model, trust_remote_code=True)
        cache[model] = (tokenizer, getattr(config, "talker_config", None))
    tokenizer, talker_config = cache[model]
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        additional_information=info,
        task_type="Base",
        tokenize_prompt=lambda text: tokenizer(text, padding=False)["input_ids"],
        codec_language_id=getattr(talker_config, "codec_language_id", None),
        spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
        estimate_ref_code_len=None,
    )


def _audio_samples(multimodal_output: Dict[str, Any]) -> tuple[np.ndarray, int]:
    import torch

    audio_data = multimodal_output["audio"]
    sample_rate_raw = multimodal_output["sr"]
    sample_rate_value = sample_rate_raw[-1] if isinstance(sample_rate_raw, list) and sample_rate_raw else sample_rate_raw
    sample_rate = int(sample_rate_value.item() if hasattr(sample_rate_value, "item") else sample_rate_value)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    samples = audio_tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
    return np.asarray(samples, dtype=np.float32), sample_rate


def _request_index(request_id: str) -> Optional[int]:
    prefix = request_id.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def main() -> int:
    state = VllmOmniWorkerState()
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        command: Dict[str, Any] = {}
        try:
            command = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                payload = handle_command(state, command)
            response = {
                "id": command.get("id"),
                "ok": True,
                "payload": payload,
            }
        except Exception as exc:
            response = {
                "id": command.get("id") if isinstance(command, dict) else None,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_stdout.flush()
        if isinstance(response.get("payload"), dict) and response["payload"].get("shutdown"):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
