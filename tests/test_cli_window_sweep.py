from ebook_tts_pipeline.cli import build_parser


def test_cli_accepts_benchmark_window_sweep_command():
    args = build_parser().parse_args(
        [
            "benchmark-window-sweep",
            "--book-root",
            "book",
            "--chapter",
            "chapter_015",
            "--start-chars",
            "100",
            "--step-chars",
            "100",
            "--max-vram-gb",
            "10",
            "--playback-speed",
            "1.0",
            "--warmup-text",
            "Test",
            "--repeat-count",
            "3",
            "--max-targets",
            "5",
        ]
    )

    assert args.command == "benchmark-window-sweep"
    assert args.book_root == "book"
    assert args.chapter == "chapter_015"
    assert args.start_chars == 100
    assert args.step_chars == 100
    assert args.max_vram_gb == 10.0
    assert args.playback_speed == 1.0
    assert args.warmup_text == "Test"
    assert args.repeat_count == 3
    assert args.max_targets == 5
