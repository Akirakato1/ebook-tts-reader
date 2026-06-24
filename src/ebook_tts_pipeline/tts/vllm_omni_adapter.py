from __future__ import annotations

import json
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ebook_tts_pipeline.config import DEFAULT_VLLM_OMNI_MODEL, DEFAULT_VLLM_OMNI_STAGE_CONFIG
from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter
from ebook_tts_pipeline.tts.wsl_paths import to_wsl_path, translate_job_paths
from ebook_tts_pipeline.tts.wsl_worker import decode_audio_item


class WslVllmOmniQwenAdapter:
    def __init__(
        self,
        book_root: Path | str,
        distro: str = "Ubuntu-24.04",
        python_path: str = "/opt/ebook-vllm-omni-venv/bin/python",
        model: str = DEFAULT_VLLM_OMNI_MODEL,
        stage_configs_path: Path | str = DEFAULT_VLLM_OMNI_STAGE_CONFIG,
        voice_model_root: Path | str = "models/qwen-tts",
        voice_python_path: str = "/opt/ebook-tts-venv/bin/python",
        voice_model_choice: str = "1.7B",
        voice_device: str = "cuda",
        voice_precision: str = "bf16",
        voice_attention: str = "auto",
        timeout_seconds: float = 600.0,
        start_process: bool = True,
        project_root: Path | str | None = None,
    ) -> None:
        self.book_root = Path(book_root)
        self.distro = str(distro)
        self.python_path = str(python_path)
        self.model = _model_argument_for_wsl(model)
        self.stage_configs_path = Path(stage_configs_path)
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
        self.timeout_seconds = float(timeout_seconds)
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
            "bash",
            "-lc",
            (
                f"cd {shlex.quote(to_wsl_path(self.project_root))} && "
                f"PYTHONPATH=src {shlex.quote(self.python_path)} "
                "-m ebook_tts_pipeline.tts.vllm_omni_worker"
            ),
        ]
        self.init_payload = {
            "book_root": to_wsl_path(self.book_root),
            "model": self.model,
            "stage_configs_path": to_wsl_path(self.stage_configs_path),
            "output_dir": to_wsl_path(self.book_root / "read_along" / "vllm_omni_tmp_audio"),
        }
        log_runtime_step(
            "vllm_omni_adapter_config",
            book_root=self.book_root,
            model=self.model,
            stage_config=self.stage_configs_path,
            voice_model_root=Path(voice_model_root),
            distro=self.distro,
            python=self.python_path,
        )
        self._voice_worker = WslQwenWorkerAdapter(
            book_root=self.book_root,
            model_root=Path(voice_model_root),
            distro=self.distro,
            python_path=voice_python_path,
            model_choice=voice_model_choice,
            device=voice_device,
            precision=voice_precision,
            attention=voice_attention,
            timeout_seconds=timeout_seconds,
            start_process=False,
        )
        if start_process:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return
            log_runtime_step(
                "vllm_omni_worker_start",
                distro=self.distro,
                python=self.python_path,
                model=self.model,
                stage_config=self.stage_configs_path,
            )
            self._process = subprocess.Popen(
                self.worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self._request("init", self.init_payload)

    def close(self) -> None:
        with self._lock:
            if self._process is not None:
                try:
                    self._request("shutdown", {})
                finally:
                    self._process.terminate()
                    self._process = None
            self._voice_worker.close()

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        voice_path = Path(voice_path)
        if voice_path.exists():
            log_runtime_step("vllm_omni_ensure_voice_cached", role_id=role_id, voice_path=voice_path)
            return voice_path
        log_runtime_step("vllm_omni_ensure_voice_generate", role_id=role_id, voice_path=voice_path)
        return self._voice_worker.ensure_voice(role_id, voice_record, voice_path)

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        yield self.generate_sentences(jobs)

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        self.start()
        log_runtime_step("vllm_omni_generate_sentences", job_count=len(jobs))
        payload = self._generate_payload(jobs)
        response = self._request("generate_sentences", payload)
        return [decode_audio_item(item) for item in response["items"]]

    def _generate_payload(self, jobs: List[Dict]) -> Dict[str, Any]:
        return {"jobs": translate_job_paths(jobs, self.book_root)}

    def _request(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("vLLM-Omni WSL worker is not running.")
        request_id = self._next_id
        self._next_id += 1
        request = {"id": request_id, "command": command, "payload": payload}
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        while True:
            line = self._process.stdout.readline()
            if not line:
                stderr = self._process.stderr.read() if self._process.stderr is not None else ""
                raise RuntimeError(f"vLLM-Omni WSL worker stopped before responding. stderr={stderr}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("id") != request_id:
                continue
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "vLLM-Omni worker failed"))
            return dict(response.get("payload") or {})


def _model_argument_for_wsl(model: Path | str) -> str:
    value = str(model)
    path = Path(value)
    if path.exists() or (len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"}):
        return to_wsl_path(path)
    return value
