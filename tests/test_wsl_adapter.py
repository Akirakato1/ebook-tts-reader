import threading
from pathlib import Path

from ebook_tts_pipeline.tts import wsl_adapter as wsl_module
from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter


def test_wsl_adapter_builds_worker_command(capsys):
    adapter = WslQwenWorkerAdapter(
        book_root=Path(r"C:\book"),
        model_root=Path(r"C:\book\models\qwen-tts"),
        distro="Ubuntu-24.04",
        python_path="/opt/ebook-tts-venv/bin/python",
        start_process=False,
    )

    assert adapter.worker_command == [
        "wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "-u",
        "root",
        "--",
        "/opt/ebook-tts-venv/bin/python",
        "-m",
        "ebook_tts_pipeline.tts.wsl_worker",
    ]
    output = capsys.readouterr().out
    assert "[ebook-tts] wsl_qwen_adapter_config" in output
    assert "model_root=C:\\book\\models\\qwen-tts" in output


def test_wsl_adapter_drains_worker_stderr_while_process_runs(monkeypatch):
    fake_process = _FakeProcess(
        ['{"id": 0, "ok": true, "payload": {"ready": true}}\n'],
        stderr_lines=["qwen noisy stderr log\n"],
    )
    monkeypatch.setattr(
        wsl_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )

    WslQwenWorkerAdapter(
        book_root=Path(r"C:\book"),
        model_root=Path(r"C:\book\models\qwen-tts"),
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
