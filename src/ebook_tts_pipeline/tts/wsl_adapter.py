from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_paths import to_wsl_path, translate_job_paths
from ebook_tts_pipeline.tts.wsl_worker import decode_audio_item


class WslQwenWorkerAdapter:
    def __init__(
        self,
        book_root: Path | str,
        model_root: Path | str,
        distro: str = "Ubuntu-24.04",
        python_path: str = "/opt/ebook-tts-venv/bin/python",
        model_choice: str = "1.7B",
        device: str = "cuda",
        precision: str = "bf16",
        attention: str = "auto",
        max_new_tokens: int = 2048,
        max_generation_block_chars: int = 0,
        max_generation_blocks_per_call: int = 0,
        cache_clear_interval: int = 8,
        streaming_text_mode: bool = True,
        performance_log_path: Optional[Path] = None,
        adaptive_memory_target_bytes: Optional[int] = None,
        timeout_seconds: float = 600.0,
        start_process: bool = True,
    ) -> None:
        self.book_root = Path(book_root)
        self.model_root = Path(model_root)
        self.distro = str(distro)
        self.python_path = str(python_path)
        self.timeout_seconds = float(timeout_seconds)
        self.role_voice_paths: Dict[str, Path] = {}
        self._lock = threading.RLock()
        self._next_id = 0
        self._process: Optional[subprocess.Popen[str]] = None
        self.worker_command = [
            "wsl.exe",
            "-d",
            self.distro,
            "-u",
            "root",
            "--",
            self.python_path,
            "-m",
            "ebook_tts_pipeline.tts.wsl_worker",
        ]
        self._init_payload = {
            "model_root": to_wsl_path(self.model_root),
            "model_choice": model_choice,
            "device": device,
            "precision": precision,
            "attention": attention,
            "max_new_tokens": int(max_new_tokens),
            "max_generation_block_chars": int(max_generation_block_chars),
            "max_generation_blocks_per_call": int(max_generation_blocks_per_call),
            "cache_clear_interval": int(cache_clear_interval),
            "streaming_text_mode": bool(streaming_text_mode),
            "performance_log_path": to_wsl_path(performance_log_path) if performance_log_path else None,
            "adaptive_memory_target_bytes": adaptive_memory_target_bytes,
        }
        log_runtime_step(
            "wsl_qwen_adapter_config",
            book_root=self.book_root,
            model_root=self.model_root,
            wsl_model_root=self._init_payload["model_root"],
            distro=self.distro,
            python=self.python_path,
            precision=precision,
            attention=attention,
        )
        if start_process:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return
            log_runtime_step(
                "wsl_qwen_worker_start",
                distro=self.distro,
                python=self.python_path,
                model_root=self.model_root,
            )
            self._process = subprocess.Popen(
                self.worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self._request("init", self._init_payload)

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            try:
                self._request("shutdown", {})
            finally:
                self._process.terminate()
                self._process = None

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        self.start()
        log_runtime_step("wsl_qwen_ensure_voice", role_id=role_id, voice_path=voice_path)
        self.role_voice_paths.setdefault(role_id, voice_path)
        payload = {
            "role_id": role_id,
            "voice_record": dict(voice_record),
            "voice_path": to_wsl_path(voice_path),
        }
        self._request("ensure_voice", payload)
        return voice_path

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        yield self.generate_sentences(jobs)

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        self.start()
        log_runtime_step("wsl_qwen_generate_sentences", job_count=len(jobs))
        translated = translate_job_paths(jobs, self.book_root)
        payload = self._request("generate_sentences", {"jobs": translated})
        return [decode_audio_item(item) for item in payload["items"]]

    def _request(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("WSL worker is not running.")
        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "command": command, "payload": payload}
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr is not None else ""
            raise RuntimeError(f"WSL worker stopped before responding. stderr={stderr}")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(f"Unexpected WSL worker response id: {response.get('id')}")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "WSL worker failed"))
        return dict(response.get("payload") or {})
