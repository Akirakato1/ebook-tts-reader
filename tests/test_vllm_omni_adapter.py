import threading
from pathlib import Path

from ebook_tts_pipeline.tts import vllm_omni_adapter as vllm_module
from ebook_tts_pipeline.tts.vllm_omni_adapter import WslVllmOmniQwenAdapter
from ebook_tts_pipeline.tts.vllm_omni_worker import _force_clean_linux_path


def test_vllm_omni_adapter_builds_worker_command_and_stable_init_payload(capsys):
    adapter = WslVllmOmniQwenAdapter(
        book_root=Path(r"C:\book"),
        distro="Ubuntu-24.04",
        python_path="/opt/ebook-vllm-omni-venv/bin/python",
        model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        stage_configs_path=Path(
            r"C:\repo\scripts\vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml"
        ),
        voice_model_root=Path(r"C:\book\models\qwen-tts"),
        voice_python_path="/opt/ebook-tts-venv/bin/python",
        project_root=Path(r"C:\repo"),
        start_process=False,
    )

    assert adapter.worker_command == [
        "wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "-u",
        "root",
        "--",
        "bash",
        "-lc",
        (
            "cd /mnt/c/repo && PYTHONPATH=src /opt/ebook-vllm-omni-venv/bin/python "
            "-m ebook_tts_pipeline.tts.vllm_omni_worker"
        ),
    ]
    assert adapter.init_payload == {
        "book_root": "/mnt/c/book",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "stage_configs_path": "/mnt/c/repo/scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml",
        "output_dir": "/mnt/c/book/read_along/vllm_omni_tmp_audio",
    }
    output = capsys.readouterr().out
    assert "[ebook-tts] vllm_omni_adapter_config" in output
    assert "model=Qwen/Qwen3-TTS-12Hz-1.7B-Base" in output
    assert "voice_model_root=C:\\book\\models\\qwen-tts" in output


def test_vllm_omni_adapter_uses_local_model_path_without_hf_resolution(tmp_path):
    model_root = tmp_path / "models" / "qwen-tts" / "Qwen3-TTS-12Hz-1.7B-Base"
    model_root.mkdir(parents=True)

    adapter = WslVllmOmniQwenAdapter(
        book_root=Path(r"C:\book"),
        model=model_root,
        stage_configs_path=Path(r"C:\repo\scripts\stable.yaml"),
        voice_model_root=Path(r"C:\book\models\qwen-tts"),
        start_process=False,
    )

    assert adapter.init_payload["model"].startswith("/mnt/")
    assert adapter.init_payload["model"].endswith(
        "/models/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base"
    )
    assert adapter.init_payload["model"] != "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_vllm_omni_worker_forces_offline_model_resolution(monkeypatch):
    monkeypatch.setenv("PATH", "WINDOWS_PATH_SENTINEL")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    _force_clean_linux_path()

    import os

    if os.name == "nt":
        assert os.environ["PATH"] == "WINDOWS_PATH_SENTINEL"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_vllm_omni_adapter_translates_voice_paths_before_generation():
    adapter = WslVllmOmniQwenAdapter(
        book_root=Path(r"C:\book"),
        stage_configs_path=Path(r"C:\repo\scripts\stable.yaml"),
        voice_model_root=Path(r"C:\book\models\qwen-tts"),
        start_process=False,
    )

    payload = adapter._generate_payload(
        [
            {
                "sentence_idx": 7,
                "unit_idx": 3,
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "text": "Hello.",
                "voice_config_path": "voices/narrator.qvp",
            }
        ]
    )

    assert payload["jobs"][0]["voice_config_path"] == "/mnt/c/book/voices/narrator.qvp"


def test_vllm_omni_adapter_skips_stdout_log_lines_before_json_response():
    adapter = WslVllmOmniQwenAdapter(
        book_root=Path(r"C:\book"),
        stage_configs_path=Path(r"C:\repo\scripts\stable.yaml"),
        voice_model_root=Path(r"C:\book\models\qwen-tts"),
        start_process=False,
    )
    adapter._process = _FakeProcess(
        [
            "INFO 06-23 vllm noisy startup log\n",
            '{"id": 0, "ok": true, "payload": {"ready": true}}\n',
        ]
    )

    assert adapter._request("init", {}) == {"ready": True}


def test_vllm_omni_adapter_drains_worker_stderr_while_process_runs(monkeypatch):
    fake_process = _FakeProcess(
        ['{"id": 0, "ok": true, "payload": {"ready": true}}\n'],
        stderr_lines=["vllm noisy stderr log\n"],
    )
    monkeypatch.setattr(
        vllm_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )

    WslVllmOmniQwenAdapter(
        book_root=Path(r"C:\book"),
        stage_configs_path=Path(r"C:\repo\scripts\stable.yaml"),
        voice_model_root=Path(r"C:\book\models\qwen-tts"),
        start_process=True,
    )

    assert fake_process.stderr.drained.wait(timeout=1.0)


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        return None


class _FakeStdout:
    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        if not self.lines:
            return ""
        return self.lines.pop(0)


class _FakeStderr:
    def __init__(self, lines=None):
        self.lines = list(lines or [])
        self.drained = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.lines:
            raise StopIteration
        self.drained.set()
        return self.lines.pop(0)

    def read(self):
        self.drained.set()
        return ""


class _FakeProcess:
    def __init__(self, stdout_lines, stderr_lines=None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStderr(stderr_lines)
