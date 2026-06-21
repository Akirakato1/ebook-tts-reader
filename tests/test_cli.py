from ebook_tts_pipeline.cli import build_parser, main
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager


def test_project_exposes_tkinter_ui_entry_point():
    pyproject = read_json_like_toml_scripts()

    assert pyproject["ebook-tts-ui"] == "ebook_tts_pipeline.ui.tk_app:main"


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


def test_cli_has_discrete_pipeline_step_commands():
    parser = build_parser()

    segment = parser.parse_args(
        ["segment-chapter", "--book-root", "books/demo", "--chapter", "chapter_001"]
    )
    annotate = parser.parse_args(
        [
            "annotate-chapter",
            "--book-root",
            "books/demo",
            "--book-title",
            "Demo",
            "--book-slug",
            "demo",
            "--chapter",
            "chapter_001",
        ]
    )
    build_tts = parser.parse_args(
        ["build-tts-script", "--book-root", "books/demo", "--chapter", "chapter_001"]
    )
    build_global = parser.parse_args(
        [
            "build-global-registry",
            "--book-root",
            "books/demo",
            "--book-title",
            "Demo",
            "--book-slug",
            "demo",
        ]
    )
    prepare = parser.parse_args(
        ["prepare-voices", "--book-root", "books/demo", "--chapter", "chapter_001", "--fake-tts"]
    )
    synthesize = parser.parse_args(
        [
            "synthesize-chapter",
            "--book-root",
            "books/demo",
            "--chapter",
            "chapter_001",
            "--fake-tts",
            "--regenerate-voices",
        ]
    )

    assert segment.command == "segment-chapter"
    assert annotate.command == "annotate-chapter"
    assert build_tts.command == "build-tts-script"
    assert build_global.command == "build-global-registry"
    assert build_global.book_title == "Demo"
    assert prepare.command == "prepare-voices"
    assert synthesize.command == "synthesize-chapter"
    assert synthesize.fake_tts is True
    assert synthesize.rebuild_tts_script is False
    assert synthesize.regenerate_voices is True


def test_cli_build_tts_script_uses_saved_annotation_without_audio(tmp_path):
    book_root = tmp_path / "demo"
    paths = _setup_narrator_book(book_root)
    write_json_atomic(
        paths.sentence_artifact("chapter_001"),
        {
            "chapter": "chapter_001",
            "source_path": "chapters/chapter_001.txt",
            "segmenter": {"name": "test", "language": "english", "version": "test"},
            "sentences": [{"idx": 0, "text": "Hello from the saved sentence."}],
        },
    )
    _write_narrator_annotation(paths)

    result = main(
        [
            "build-tts-script",
            "--book-root",
            str(book_root),
            "--chapter",
            "chapter_001",
        ]
    )

    assert result == 0
    script = read_json(paths.tts_script("chapter_001"))
    assert script["job_count"] == 1
    assert script["jobs"][0]["role"] == "Narrator"
    assert paths.qwen_script("chapter_001").read_text(encoding="utf-8").strip() == (
        "Narrator: Hello from the saved sentence."
    )
    assert not paths.chapter_audio("chapter_001").exists()


def test_cli_synthesize_chapter_reuses_saved_tts_script_without_sentence_segments(tmp_path):
    book_root = tmp_path / "demo"
    paths = _setup_narrator_book(book_root)
    _write_narrator_annotation(paths)
    write_json_atomic(
        paths.tts_script("chapter_001"),
        {
            "chapter": "chapter_001",
            "job_count": 1,
            "window_count": 1,
            "qwen_dialogue_text": "Narrator: Hello from saved TTS.",
            "jobs": [
                {
                    "sentence_idx": 0,
                    "role": "Narrator",
                    "role_id": "narrator",
                    "type": "narration",
                    "text": "Hello from saved TTS.",
                    "voice_config_path": None,
                }
            ],
            "windows": [],
        },
    )

    result = main(
        [
            "synthesize-chapter",
            "--book-root",
            str(book_root),
            "--chapter",
            "chapter_001",
            "--fake-tts",
        ]
    )

    assert result == 0
    assert paths.voice_qvp("narrator").exists()
    assert paths.chapter_audio("chapter_001").exists()
    assert paths.chapter_timeline("chapter_001").exists()
    assert not paths.sentence_artifact("chapter_001").exists()


def _setup_narrator_book(book_root):
    paths = BookPaths(book_root)
    RegistryManager(paths).initialize_if_missing(book_title="Demo", book_slug="demo")
    return paths


def _write_narrator_annotation(paths):
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "roles": ["Narrator"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0]],
        },
    )


def read_json_like_toml_scripts():
    scripts = {}
    in_scripts = False
    for line in open("pyproject.toml", encoding="utf-8"):
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and stripped.startswith("["):
            break
        if in_scripts and "=" in stripped:
            key, value = stripped.split("=", 1)
            scripts[key.strip()] = value.strip().strip('"')
    return scripts
