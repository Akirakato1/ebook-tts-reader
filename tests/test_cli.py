from ebook_tts_pipeline.cli import build_parser


def test_cli_has_run_chapter_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-chapter",
            "--book-root",
            "books/demo",
            "--book-title",
            "Demo",
            "--book-slug",
            "demo",
            "--chapter",
            "chapter_001",
            "--fake-tts",
        ]
    )

    assert args.command == "run-chapter"
    assert args.book_root == "books/demo"
    assert args.chapter == "chapter_001"
    assert args.fake_tts is True
