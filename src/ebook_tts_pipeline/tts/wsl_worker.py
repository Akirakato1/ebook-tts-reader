from __future__ import annotations

import base64
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


def encode_audio_item(item: GeneratedSentenceAudio) -> Dict[str, Any]:
    samples = np.asarray(item.samples, dtype=np.float32)
    return {
        "sentence_idx": item.sentence_idx,
        "unit_idx": item.unit_idx,
        "role": item.role,
        "speech_type": item.speech_type,
        "sample_rate": item.sample_rate,
        "voice_config_path": item.voice_config_path,
        "pause_after_ms": item.pause_after_ms,
        "dtype": "float32",
        "shape": list(samples.shape),
        "samples_b64": base64.b64encode(samples.tobytes()).decode("ascii"),
    }


def decode_audio_item(payload: Dict[str, Any]) -> GeneratedSentenceAudio:
    samples = np.frombuffer(base64.b64decode(payload["samples_b64"]), dtype=np.float32).copy()
    samples = samples.reshape(tuple(payload.get("shape") or [len(samples)]))
    return GeneratedSentenceAudio(
        sentence_idx=int(payload["sentence_idx"]),
        unit_idx=int(payload["unit_idx"]) if payload.get("unit_idx") is not None else None,
        role=str(payload["role"]),
        speech_type=str(payload["speech_type"]),
        samples=samples,
        sample_rate=int(payload["sample_rate"]),
        pause_after_ms=payload.get("pause_after_ms"),
        voice_config_path=payload.get("voice_config_path"),
    )


class WorkerState:
    def __init__(self) -> None:
        self.adapter: Optional[QwenTtsAdapter] = None

    def init(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.adapter = QwenTtsAdapter(
            model_root=str(payload["model_root"]),
            model_choice=str(payload.get("model_choice", "1.7B")),
            device=str(payload.get("device", "cuda")),
            precision=str(payload.get("precision", "bf16")),
            attention=str(payload.get("attention", "auto")),
            max_new_tokens=int(payload.get("max_new_tokens", 2048)),
            max_generation_block_chars=int(payload.get("max_generation_block_chars", 0)),
            max_generation_blocks_per_call=int(payload.get("max_generation_blocks_per_call", 0)),
            cache_clear_interval=int(payload.get("cache_clear_interval", 8)),
            streaming_text_mode=bool(payload.get("streaming_text_mode", True)),
            performance_log_path=Path(payload["performance_log_path"]) if payload.get("performance_log_path") else None,
            adaptive_memory_target_bytes=(
                int(payload["adaptive_memory_target_bytes"])
                if payload.get("adaptive_memory_target_bytes") is not None
                else None
            ),
        )
        return {"initialized": True}

    def ensure_voice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapter = self._require_adapter()
        path = adapter.ensure_voice(
            role_id=str(payload["role_id"]),
            voice_record=dict(payload["voice_record"]),
            voice_path=Path(payload["voice_path"]),
            sample_path=Path(payload["sample_path"]) if payload.get("sample_path") else None,
            reference_text=str(payload["reference_text"]) if payload.get("reference_text") is not None else None,
        )
        return {"voice_path": str(path)}

    def generate_sentences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapter = self._require_adapter()
        jobs = []
        role_voice_paths = {}
        for job in payload["jobs"]:
            item = dict(job)
            voice_path = str(item.pop("voice_config_path", "") or "")
            role_id = str(item.get("role_id", ""))
            role = str(item.get("role", ""))
            if voice_path:
                path = Path(voice_path)
                if role_id:
                    role_voice_paths[role_id] = path
                if role:
                    role_voice_paths[role] = path
            jobs.append(item)
        adapter.role_voice_paths.update(role_voice_paths)
        return {"items": [encode_audio_item(item) for item in adapter.generate_sentences(jobs)]}

    def shutdown(self) -> Dict[str, Any]:
        self.adapter = None
        return {"shutdown": True}

    def _require_adapter(self) -> QwenTtsAdapter:
        if self.adapter is None:
            raise RuntimeError("Worker has not been initialized.")
        return self.adapter


def handle_command(state: WorkerState, command: Dict[str, Any]) -> Dict[str, Any]:
    name = str(command.get("command", ""))
    payload = dict(command.get("payload", {}))
    if name == "init":
        return state.init(payload)
    if name == "ensure_voice":
        return state.ensure_voice(payload)
    if name == "generate_sentences":
        return state.generate_sentences(payload)
    if name == "shutdown":
        return state.shutdown()
    raise ValueError(f"Unknown worker command: {name}")


def main() -> int:
    state = WorkerState()
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
