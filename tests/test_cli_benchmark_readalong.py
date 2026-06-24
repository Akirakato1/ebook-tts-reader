from pathlib import Path

from ebook_tts_pipeline import cli as cli_module
from ebook_tts_pipeline.cli import _resolve_vllm_omni_model, build_parser
from ebook_tts_pipeline.config import PipelineConfig


def test_cli_accepts_benchmark_readalong_command():
    args = build_parser().parse_args(
        [
            "benchmark-readalong",
            "--book-root",
            "book",
            "--chapter",
            "chapter_015",
            "--start-unit",
            "0",
            "--unit-count",
            "5",
            "--target-buffer-seconds",
            "10",
        ]
    )

    assert args.command == "benchmark-readalong"
    assert args.book_root == "book"
    assert args.chapter == "chapter_015"
    assert args.start_unit == 0
    assert args.unit_count == 5
    assert args.target_buffer_seconds == 10.0


def test_cli_resolves_default_vllm_omni_model_to_local_qwen_folder(tmp_path):
    model_root = tmp_path / "models" / "qwen-tts"
    base_model = model_root / "Qwen3-TTS-12Hz-1.7B-Base"
    base_model.mkdir(parents=True)
    config = PipelineConfig(book_root=str(tmp_path / "book"), qwen_model_root=str(model_root))

    resolved = _resolve_vllm_omni_model(config)

    assert resolved == base_model


def test_cli_resolves_relative_vllm_omni_model_to_absolute_local_folder(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "qwen-tts"
    base_model = model_root / "Qwen3-TTS-12Hz-1.7B-Base"
    base_model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    config = PipelineConfig(book_root="book", qwen_model_root="models/qwen-tts")

    resolved = _resolve_vllm_omni_model(config)

    assert resolved == base_model.resolve()


def test_cli_wsl_adapter_receives_absolute_model_root(tmp_path, monkeypatch, capsys):
    captured = {}

    class DummyWslAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "WslQwenWorkerAdapter", DummyWslAdapter)
    config = PipelineConfig(
        book_root=str(tmp_path / "book"),
        tts_backend="wsl",
        qwen_model_root="models/qwen-tts",
    )

    cli_module._build_qwen_adapter(config)

    assert Path(captured["model_root"]).is_absolute()
    output = capsys.readouterr().out
    assert "[ebook-tts] build_tts_adapter" in output
    assert "backend=wsl" in output
    assert str(captured["model_root"]) in output


def test_cli_vllm_adapter_receives_absolute_voice_model_root(tmp_path, monkeypatch, capsys):
    captured = {}

    class DummyVllmAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "WslVllmOmniQwenAdapter", DummyVllmAdapter)
    config = PipelineConfig(
        book_root=str(tmp_path / "book"),
        tts_backend="wsl-vllm-omni",
        qwen_model_root="models/qwen-tts",
    )

    cli_module._build_qwen_adapter(config)

    assert Path(captured["voice_model_root"]).is_absolute()
    output = capsys.readouterr().out
    assert "[ebook-tts] build_tts_adapter" in output
    assert "backend=wsl-vllm-omni" in output
    assert str(captured["voice_model_root"]) in output
