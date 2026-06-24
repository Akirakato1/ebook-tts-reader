from pathlib import Path

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
