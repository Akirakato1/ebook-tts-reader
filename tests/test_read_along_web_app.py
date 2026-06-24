from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ebook_tts_pipeline.epub_ingestion import EpubExtractResult
from ebook_tts_pipeline.ingestion import SentenceSegmenter
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.ui.web_app import build_parser, create_server


def test_web_app_parser_accepts_notebook_style_options():
    parser = build_parser()

    defaults = parser.parse_args(["--no-open"])

    assert defaults.launch_root == "."
    assert defaults.book_root is None
    assert defaults.open_browser is False

    args = parser.parse_args(
        [
            "--book-root",
            "books/demo",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--fake-tts",
            "--no-open",
        ]
    )

    assert args.book_root == "books/demo"
    assert args.launch_root == "."
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert args.fake_tts is True
    assert args.open_browser is False


def test_pyproject_exposes_short_readalongweb_command():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert re.search(r'^readalongweb\s*=\s*"ebook_tts_pipeline\.ui\.web_app:main"', text, re.MULTILINE)
    assert re.search(r'^ebook-tts-readalong-web\s*=\s*"ebook_tts_pipeline\.ui\.web_app:main"', text, re.MULTILINE)


def test_library_api_discovers_books_and_hides_non_book_dirs(tmp_path):
    _write_demo_book(tmp_path, name="alpha", title="Alpha")
    _write_demo_book(tmp_path, name="beta", title="Beta")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "random.txt").write_text("not a book", encoding="utf-8")
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        library = _get_json(base_url + "/api/library")

        assert library["ok"] is True
        assert library["mode"] == "library"
        assert library["active_book"] is None
        assert [book["slug"] for book in library["books"]] == ["alpha", "beta"]
        assert [book["title"] for book in library["books"]] == ["Alpha", "Beta"]
        assert all(book["chapter_count"] == 1 for book in library["books"])
        assert all(book["annotation_count"] == 1 for book in library["books"])
        assert all(book["status_label"] == "Annotated" for book in library["books"])
        assert all("1/1 annotated" in book["status_detail"] for book in library["books"])
        assert all(book["voice_count"] == 0 for book in library["books"])
        assert all(book["voice_total"] == 1 for book in library["books"])
    finally:
        _stop_server(server, thread)


def test_library_voice_stats_count_registry_characters_with_qvp_and_sample(tmp_path):
    paths = _write_demo_book(tmp_path, name="voices", title="Voices Book")
    registry = read_json(paths.registry)
    registry["characters"]["callie_adult"] = {
        "role_id": "callie_adult",
        "profile_id": "callie_adult",
        "person_id": "callie",
        "display_name": "Callie",
        "age_stage": "adult",
        "aliases": [],
        "voice_config_path": "voices/callie_adult.qvp",
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_identity": {"seed": 3, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
    }
    write_json_atomic(paths.registry, registry)
    (paths.root / "voices" / "_samples").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "_samples" / "leigh_adult.wav").write_bytes(b"sample")
    (paths.root / "voices" / "callie_adult.qvp").write_bytes(b"voice")
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        library = _get_json(base_url + "/api/library")
        book = library["books"][0]

        assert book["voice_count"] == 1
        assert book["voice_total"] == 2
        assert "1/2 voices" in book["status_detail"]
    finally:
        _stop_server(server, thread)


def test_direct_book_launch_root_opens_reader_without_selection(tmp_path):
    paths = _write_demo_book(tmp_path, name="direct", title="Direct Book")
    server, thread, base_url = _start_test_server_for_root(paths.root)
    try:
        library = _get_json(base_url + "/api/library")
        state = _get_json(base_url + "/api/state")

        assert library["mode"] == "book"
        assert library["active_book"]["slug"] == "direct"
        assert state["chapters"][0]["title"] == "Chapter One"
    finally:
        _stop_server(server, thread)


def test_library_selection_switches_active_book(tmp_path):
    _write_demo_book(tmp_path, name="alpha", title="Alpha", ready_for_tts=True)
    _write_demo_book(tmp_path, name="beta", title="Beta", ready_for_tts=True)
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        selected = _post_json(base_url + "/api/library/select", {"slug": "beta"})
        state = _get_json(base_url + "/api/state")

        assert selected["ok"] is True
        assert selected["active_book"]["slug"] == "beta"
        assert state["active_book"]["slug"] == "beta"
        assert state["chapters"][0]["chapter"] == "chapter_001"
    finally:
        _stop_server(server, thread)


def test_library_delete_removes_book_and_clears_active_selection(tmp_path):
    _write_demo_book(tmp_path, name="alpha", title="Alpha", ready_for_tts=True)
    beta_paths = _write_demo_book(tmp_path, name="beta", title="Beta", ready_for_tts=True)
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        _post_json(base_url + "/api/library/select", {"slug": "beta"})

        deleted = _post_json(base_url + "/api/library/delete", {"slug": "beta"})

        assert deleted["ok"] is True
        assert deleted["deleted_slug"] == "beta"
        assert deleted["active_book"] is None
        assert not beta_paths.root.exists()
        assert (tmp_path / "alpha").exists()
        assert [book["slug"] for book in deleted["library"]["books"]] == ["alpha"]

        error = _post_json(
            base_url + "/api/library/delete",
            {"slug": "beta"},
            expect_status=400,
        )

        assert error["ok"] is False
        assert "Book not found" in error["error"]
    finally:
        _stop_server(server, thread)


def test_add_book_creates_pending_entry_without_initializing_or_opening(tmp_path):
    source = tmp_path / "source.epub"
    source.write_bytes(b"fake epub bytes")
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        added = _post_json(
            base_url + "/api/library/add-book",
            {
                "epub_path": str(source),
                "title": "New Book",
                "author": "Karin Slaughter",
                "slug": "new-book",
            },
        )
        library = _get_json(base_url + "/api/library")

        assert added["ok"] is True
        assert added["book"]["slug"] == "new-book"
        assert added["book"]["author"] == "Karin Slaughter"
        assert added["book"]["status_key"] == "fresh_added"
        assert added["book"]["action_key"] == "initialize"
        assert added["book"]["open_enabled"] is False
        assert added["active_book"] is None
        assert (tmp_path / "new-book" / "_source" / "original.epub").read_bytes() == b"fake epub bytes"
        manifest = read_json(tmp_path / "new-book" / "readalong_book.json")
        assert manifest["author"] == "Karin Slaughter"
        assert not (tmp_path / "new-book" / "chapters").exists()
        assert library["books"][0]["slug"] == "new-book"
        assert library["books"][0]["author"] == "Karin Slaughter"
        assert library["books"][0]["last_read_label"] == "Not started"
    finally:
        _stop_server(server, thread)


def test_add_book_upload_creates_pending_entry_without_opening(tmp_path):
    upload_bytes = b"uploaded epub bytes"
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        added = _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Uploaded Book", "author": "Karin Slaughter", "slug": "uploaded-book"},
            files={"epub": ("uploaded.epub", "application/epub+zip", upload_bytes)},
        )
        library = _get_json(base_url + "/api/library")

        assert added["ok"] is True
        assert added["book"]["slug"] == "uploaded-book"
        assert added["book"]["author"] == "Karin Slaughter"
        assert added["book"]["status_label"] == "Freshly added"
        assert added["book"]["action_label"] == "Initialize Book"
        assert added["book"]["open_enabled"] is False
        assert added["active_book"] is None
        assert (tmp_path / "uploaded-book" / "_source" / "original.epub").read_bytes() == upload_bytes
        assert not (tmp_path / "uploaded-book" / "chapters").exists()
        assert library["books"][0]["status_key"] == "fresh_added"
        assert library["books"][0]["action_key"] == "initialize"
        assert read_json(tmp_path / "uploaded-book" / "readalong_book.json")["author"] == "Karin Slaughter"
    finally:
        _stop_server(server, thread)


def test_book_lifecycle_requires_explicit_steps_before_open_and_tracks_last_read(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
        pipeline_factory=_fake_lifecycle_pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        added = _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Lifecycle Book", "author": "Karin Slaughter", "slug": "lifecycle-book"},
            files={"epub": ("lifecycle-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )

        assert added["book"]["status_key"] == "fresh_added"
        assert added["book"]["action_key"] == "initialize"
        assert added["book"]["open_enabled"] is False
        assert _post_json(
            base_url + "/api/library/select",
            {"slug": "lifecycle-book"},
            expect_status=400,
        )["error"].startswith("Initialize Book")

        initialized = _post_json(base_url + "/api/library/initialize", {"slug": "lifecycle-book"})

        assert initialized["book"]["status_key"] == "initialized"
        assert initialized["book"]["action_key"] == "build_registry"
        assert initialized["book"]["open_enabled"] is False
        assert (tmp_path / "lifecycle-book" / "chapters" / "chapter_001.txt").exists()
        assert (tmp_path / "lifecycle-book" / "sentence_segments" / "chapter_001.sentences.json").exists()
        assert read_json(tmp_path / "lifecycle-book" / "registry.json")["book"]["author"] == "Karin Slaughter"

        registry = _post_json(base_url + "/api/library/build-registry", {"slug": "lifecycle-book"})

        assert registry["book"]["status_key"] == "registry_ready"
        assert registry["book"]["action_key"] == "annotate"
        assert registry["book"]["open_enabled"] is False

        started_annotation = _post_json(base_url + "/api/library/annotate", {"slug": "lifecycle-book"})
        annotated = _wait_for_job(base_url, started_annotation["job"]["job_id"])

        assert annotated["book"]["status_key"] == "annotated"
        assert annotated["book"]["action_key"] == "prepare_voices"
        assert annotated["book"]["action_label"] == "Generate Voices"
        assert annotated["book"]["open_enabled"] is False
        assert annotated["book"]["resume_annotation_enabled"] is False
        assert (tmp_path / "lifecycle-book" / "annotations" / "chapter_001.annotation.json").exists()
        assert (tmp_path / "lifecycle-book" / "read_along" / "chapter_001.units.json").exists()
        assert _post_json(
            base_url + "/api/library/select",
            {"slug": "lifecycle-book"},
            expect_status=400,
        )["error"].startswith("Generate Voices")

        review = _get_json(base_url + "/api/registry?slug=lifecycle-book")

        assert review["ok"] is True
        assert [entry["role_id"] for entry in review["review"]["entries"]] == ["leigh_adult"]

        started_voices = _post_json(base_url + "/api/library/prepare-voices", {"slug": "lifecycle-book"})
        assert started_voices["job"]["action_key"] == "prepare_voices"
        prepared = _wait_for_job(base_url, started_voices["job"]["job_id"])

        assert prepared["book"]["status_key"] == "voices_ready"
        assert prepared["book"]["action_key"] == "review_registry"
        assert prepared["book"]["action_label"] == "Review Voices"
        assert prepared["book"]["open_enabled"] is True
        assert prepared["book"]["voice_count"] == 1
        assert prepared["book"]["voice_total"] == 1
        assert (tmp_path / "lifecycle-book" / "voices" / "_samples" / "leigh_adult.wav").exists()

        opened = _post_json(base_url + "/api/library/select", {"slug": "lifecycle-book"})

        assert opened["active_book"]["slug"] == "lifecycle-book"

        _post_json(
            base_url + "/api/session/start",
            {
                "chapter": "chapter_001",
                "start_unit_id": 0,
                "settings": {
                    "playback_speed": 1.0,
                    "generation_mode": "balanced",
                    "buffer_limit": 1,
                    "target_buffer_seconds": 0.1,
                    "start_buffer_seconds": 0.1,
                    "max_buffer_seconds": 0.2,
                    "max_buffer_units": 1,
                    "narrator_voice_type": "current",
                },
            },
        )
        library = _get_json(base_url + "/api/library")

        assert library["active_book"]["last_read"]["chapter"] == "chapter_001"
        assert library["active_book"]["last_read"]["unit_id"] == 0
        assert "chapter_001" in library["active_book"]["last_read_label"]
    finally:
        _stop_server(server, thread)


def test_annotate_book_starts_background_job_and_reports_progress(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
        pipeline_factory=_fake_lifecycle_pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Lifecycle Book", "slug": "lifecycle-book"},
            files={"epub": ("lifecycle-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "lifecycle-book"})
        _post_json(base_url + "/api/library/build-registry", {"slug": "lifecycle-book"})

        started = _post_json(base_url + "/api/library/annotate", {"slug": "lifecycle-book"})

        assert started["ok"] is True
        assert started["job"]["action_key"] == "annotate"
        assert started["job"]["status"] in {"queued", "running", "completed"}
        assert started["book"]["status_key"] == "annotating"

        final = _wait_for_job(base_url, started["job"]["job_id"])

        assert final["job"]["status"] == "completed"
        assert final["job"]["completed"] == 1
        assert final["job"]["total"] == 1
        assert final["book"]["status_key"] == "annotated"
        assert final["book"]["action_key"] == "prepare_voices"
        assert final["book"]["action_label"] == "Generate Voices"
        assert final["book"]["resume_annotation_enabled"] is False
    finally:
        _stop_server(server, thread)


def test_failed_annotate_job_reports_failed_chapter_and_keeps_book_closed(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_TwoChapterExtractor(),
        pipeline_factory=_failing_second_chapter_pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Failing Book", "slug": "failing-book"},
            files={"epub": ("failing-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "failing-book"})
        _post_json(base_url + "/api/library/build-registry", {"slug": "failing-book"})

        started = _post_json(base_url + "/api/library/annotate", {"slug": "failing-book"})
        final = _wait_for_job(base_url, started["job"]["job_id"])

        assert final["job"]["status"] == "failed"
        assert final["job"]["failed_chapter"] == "chapter_002"
        assert "Annotation failed at chapter_002" in final["job"]["error"]
        assert final["book"]["open_enabled"] is False
        assert final["book"]["status_key"] == "registry_ready"
    finally:
        _stop_server(server, thread)


def test_retry_annotation_resumes_at_failed_chapter_without_rebuilding_previous_units(tmp_path):
    pipeline_factory = _FailOnceSecondChapterPipelineFactory()
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_TwoChapterExtractor(),
        pipeline_factory=pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Retry Book", "slug": "retry-book"},
            files={"epub": ("retry-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "retry-book"})
        _post_json(base_url + "/api/library/build-registry", {"slug": "retry-book"})

        first = _post_json(base_url + "/api/library/annotate", {"slug": "retry-book"})
        failed = _wait_for_job(base_url, first["job"]["job_id"])

        assert failed["job"]["status"] == "failed"
        assert failed["job"]["failed_chapter"] == "chapter_002"
        assert pipeline_factory.annotate_calls == ["chapter_001", "chapter_002"]
        assert pipeline_factory.unit_calls == ["chapter_001"]

        second = _post_json(base_url + "/api/library/annotate", {"slug": "retry-book"})
        completed = _wait_for_job(base_url, second["job"]["job_id"])

        assert completed["job"]["status"] == "completed"
        assert completed["book"]["status_key"] == "annotated"
        assert completed["book"]["action_key"] == "prepare_voices"
        assert pipeline_factory.annotate_calls == ["chapter_001", "chapter_002", "chapter_002"]
        assert pipeline_factory.unit_calls == ["chapter_001", "chapter_002"]
    finally:
        _stop_server(server, thread)


def test_registry_save_invalidates_voice_readiness_and_returns_updated_review_payload(tmp_path):
    paths = _write_demo_book(tmp_path, name="ready-book", title="Ready Book")
    (paths.root / "voices").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "narrator.qvp").write_bytes(b"voice")
    (paths.root / "voices" / "leigh_adult.qvp").write_bytes(b"voice")
    write_json_atomic(
        paths.root / "readalong_book.json",
        {
            "schema": "readalong_book_v1",
            "title": "Ready Book",
            "slug": "ready-book",
            "source_epub_path": "_source/original.epub",
            "original_filename": "ready.epub",
            "stages": {
                "source_added": True,
                "initialized": True,
                "global_registry": True,
                "annotated": True,
                "registry_reviewed": True,
                "voices_ready": True,
            },
        },
    )
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        before = _get_json(base_url + "/api/library")
        assert before["books"][0]["action_key"] == "review_registry"

        saved = _post_json(
            base_url + "/api/registry/save-character",
            {
                "slug": "ready-book",
                "role_id": "leigh_adult",
                "fields": {
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": "direct, careful",
                    "race_or_ethnicity": "Japanese",
                    "accent": "Tokyo",
                    "occupation": "lawyer",
                    "aliases": "Leigh",
                },
            },
        )

        assert saved["ok"] is True
        assert saved["book"]["status_key"] == "annotated"
        assert saved["book"]["action_key"] == "prepare_voices"
        assert saved["book"]["action_label"] == "Generate Voices"
        assert saved["review"]["entries"][0]["fields"]["race_or_ethnicity"] == "Japanese"
        assert saved["review"]["entries"][0]["fields"]["accent"] == "Tokyo"
        assert (paths.root / "voices" / "_samples" / "leigh_adult.wav").exists()
        reloaded = read_json(paths.registry)
        assert reloaded["characters"]["leigh_adult"]["voice_config_hash"]
        error = _post_json(base_url + "/api/library/select", {"slug": "ready-book"}, expect_status=400)
        assert "Generate Voices" in error["error"]
    finally:
        _stop_server(server, thread)


def test_registry_generate_sample_endpoint_serves_wav_preview(tmp_path):
    paths = _write_demo_book(tmp_path, name="sample-book", title="Sample Book")
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        generated = _post_json(
            base_url + "/api/registry/generate-sample",
            {"slug": "sample-book", "role_id": "leigh_adult"},
        )

        assert generated["ok"] is True
        assert generated["sample"]["role_id"] == "leigh_adult"
        assert generated["sample"]["sample_url"].startswith("/api/registry/sample/leigh_adult.wav?slug=sample-book")
        sample_path = paths.root / "voices" / "_samples" / "leigh_adult.wav"
        assert sample_path.exists()

        wav = _get_bytes(base_url + generated["sample"]["sample_url"])

        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav[:16]
    finally:
        _stop_server(server, thread)


def test_initialized_book_cannot_open_before_registry_or_annotation(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Fresh Book", "slug": "fresh-book"},
            files={"epub": ("fresh-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "fresh-book"})

        error = _post_json(
            base_url + "/api/library/select",
            {"slug": "fresh-book"},
            expect_status=400,
        )

        assert "Build Registry" in error["error"]
    finally:
        _stop_server(server, thread)


def test_unopened_book_start_session_requires_selecting_ready_book(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Fresh Book", "slug": "fresh-book"},
            files={"epub": ("fresh-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )
        _post_json(base_url + "/api/library/initialize", {"slug": "fresh-book"})

        error = _post_json(
            base_url + "/api/session/start",
            {
                "chapter": "chapter_001",
                "start_unit_id": 0,
                "settings": {
                    "playback_speed": 1.0,
                    "generation_mode": "balanced",
                    "buffer_limit": 2,
                    "target_buffer_seconds": 20,
                    "start_buffer_seconds": 20,
                    "max_buffer_seconds": 40,
                    "max_buffer_units": 32,
                    "narrator_voice_type": "male",
                },
            },
            expect_status=400,
        )

        assert "Select" in error["error"]
    finally:
        _stop_server(server, thread)


def test_add_book_upload_truncates_overlong_folder_slug(tmp_path):
    long_slug = "a" * 180
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        added = _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Very Long Folder Book", "slug": long_slug},
            files={"epub": ("very-long-folder-book.epub", "application/epub+zip", b"uploaded epub bytes")},
        )

        assert added["active_book"] is None
        slug = added["book"]["slug"]
        assert len(slug) == 80
        assert slug == "a" * 80
        assert (tmp_path / slug / "_source" / "original.epub").exists()
    finally:
        _stop_server(server, thread)


def test_add_book_upload_requires_epub_file(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        error = _post_multipart(
            base_url + "/api/library/add-book",
            fields={"title": "Uploaded Book", "slug": "uploaded-book"},
            files={},
            expect_status=400,
        )

        assert error["ok"] is False
        assert "epub" in error["error"].lower()
        assert not (tmp_path / "uploaded-book").exists()
    finally:
        _stop_server(server, thread)


def test_add_book_requires_epub_path(tmp_path):
    server = create_server(
        launch_root=tmp_path,
        host="127.0.0.1",
        port=0,
        fake_tts=True,
        extractor=_FakeExtractor(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        error = _post_json(
            base_url + "/api/library/add-book",
            {"epub_path": "", "title": "New Book", "slug": "new-book"},
            expect_status=400,
        )

        assert error["ok"] is False
        assert "epub_path" in error["error"]
        assert not (tmp_path / "new-book").exists()
    finally:
        _stop_server(server, thread)


def test_home_page_serves_clean_reader_shell(tmp_path):
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        response = _get_text(base_url + "/")

        assert "Read Along" in response
        assert "reader-text" in response
        assert "Start Session" in response
        assert 'id="speed" type="number" min="1" max="4"' in response
        assert 'id="target-buffer" type="number" min="1" max="120"' in response
        assert 'id="add-epub-file" type="file"' in response
        assert 'accept=".epub,application/epub+zip"' in response
        assert 'id="add-title" type="text" maxlength="120"' in response
        assert 'id="add-author" type="text" maxlength="120"' in response
        assert 'id="add-slug" type="text" maxlength="80"' in response
        assert response.index('class="add-book"') < response.index('id="book-list"')
        assert 'id="library-status"' in response
        assert ".add-book label {" in response
        assert "min-width: 0;" in response
        assert "text-overflow: ellipsis;" in response
        assert "Uploading book..." in response
        assert "Initializing book..." in response
        assert "book.status_label" in response
        assert "Initialize Book" in response
        assert "Build Registry" in response
        assert "Annotate Book" in response
        assert "Review Voices" in response
        assert "Generate Voices" in response
        assert 'className = "delete-book"' in response
        assert "book-list-header" in response
        assert "book-cell book-title-cell" in response
        assert "book-author-line" in response
        assert "book-cell book-status-cell" in response
        assert "book-cell book-chapters-cell" in response
        assert "book-cell book-annotated-cell" in response
        assert "book-cell book-units-cell" in response
        assert "book-cell book-voices-cell" in response
        assert "book-cell book-audio-cell" in response
        assert "book-cell book-last-read-cell" in response
        assert "book-action-cell" in response
        assert "--book-grid-columns:" in response
        assert response.count("grid-template-columns: var(--book-grid-columns);") >= 2
        assert "minmax(120px, 0.8fr) 132px" in response
        assert "border: 1px solid transparent;" in response
        assert "minmax(120px, 0.8fr) auto" not in response
        assert "truncateBookTitle" in response
        assert "title.length <= 40" in response
        assert 'data.append("author", els.addAuthor.value)' in response
        assert "book.status_detail + \" | Last read: \"" not in response
        assert "/api/library/delete" in response
        assert "confirm(" in response
        assert "border-radius: 8px;" in response
        assert "runBookAction" in response
        assert "spinner" in response
        assert "Process Book before starting read-along" in response
        assert 'id="registry-panel"' in response
        assert ".registry-panel" in response
        assert ".registry-card" in response
        assert ".registry-grid" in response
        assert "Saving registry and regenerating sample..." in response
        assert "Sample ready." in response
        assert "Review Voices" in response
        assert "openBookFromTitle" in response
        assert "book.open_enabled" in response
        assert "pollLibraryJob" in response
        assert "/api/library/job-status?job_id=" in response
        assert "state.activeJobs" in response
        assert "finishLibraryJob(jobId, payload);" in response
        assert "delete state.activeJobs[jobId];" in response
        assert "renderLibrary(payload.library);" in response
        assert "latestLibrary ||" in response
        assert "setLibraryStatus(error.message)" in response
        assert "finally {" in response
        assert "Annotating chapter" in response
        assert "resume_annotation_enabled" in response
        assert "resumeAnnotation" in response
        assert "Resume Annotation" in response
        assert "compactStatusText" in response
        assert "overflow-wrap: anywhere;" in response
        assert "max-height: 76px;" in response
        assert "/api/registry/save-character" in response
        assert "entry.sample_url" in response
        assert 'url.searchParams.set("slug", state.registryBook.slug);' in response
        assert "Run Generate Voices first." in response
        assert "/api/library/prepare-voices" in response
        assert "lockControls(Boolean(payload.session_active));" in response
        assert "window.readAlongApp" in response
    finally:
        _stop_server(server, thread)


def test_web_api_serves_chapter_and_bounded_session_audio(tmp_path):
    paths = _write_demo_book(tmp_path, ready_for_tts=True)
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        state = _get_json(base_url + "/api/state")

        assert state["ok"] is True
        assert state["settings"]["buffer_limit"] == 2
        assert state["chapters"][0]["chapter"] == "chapter_001"

        chapter = _get_json(base_url + "/api/chapter/chapter_001")

        assert chapter["ok"] is True
        assert chapter["chapter"] == "chapter_001"
        assert chapter["text"] == paths.chapter_text("chapter_001").read_text(encoding="utf-8")
        assert [unit["role_id"] for unit in chapter["units"]] == [
            "narrator",
            "leigh_adult",
            "narrator",
        ]

        started = _post_json(
            base_url + "/api/session/start",
            {
                "chapter": "chapter_001",
                "start_unit_id": 0,
                "settings": {
                    "playback_speed": 1.25,
                    "generation_mode": "balanced",
                    "buffer_limit": 2,
                    "target_buffer_seconds": 0.1,
                    "start_buffer_seconds": 0.1,
                    "max_buffer_seconds": 0.2,
                    "max_buffer_units": 4,
                    "narrator_voice_type": "current",
                },
            },
        )

        assert started["ok"] is True
        assert started["chapter"] == "chapter_001"
        assert started["ready_playback_seconds"] >= 0.1
        assert started["target_buffer_seconds"] == 0.1
        assert [item["unit_id"] for item in started["ready"]]
        assert all(item["audio_url"].endswith(".wav") for item in started["ready"])

        wav = _get_bytes(base_url + started["ready"][0]["audio_url"])

        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav[:16]

        advanced = _post_json(base_url + "/api/session/advance", {})

        assert advanced["ok"] is True
        assert "ready_playback_seconds" in advanced
        assert [item["unit_id"] for item in advanced["ready"]]

        session_dir = server.app_state.session.session_dir
        assert session_dir.exists()
        assert not list(session_dir.glob("*.wav"))

        ended = _post_json(base_url + "/api/session/end", {})

        assert ended["ok"] is True
        assert not session_dir.exists()
        assert server.app_state.session is None
    finally:
        _stop_server(server, thread)


def test_web_api_persists_manual_reading_position_and_preselects_it(tmp_path):
    paths = _write_demo_book(tmp_path, ready_for_tts=True)
    server, thread, base_url = _start_test_server_for_root(tmp_path)
    try:
        updated = _post_json(
            base_url + "/api/reading-position",
            {"slug": "book", "chapter": "chapter_001", "unit_id": 2},
        )

        assert updated["ok"] is True
        assert updated["last_read"]["chapter"] == "chapter_001"
        assert updated["last_read"]["unit_id"] == 2

        manifest = read_json(paths.root / "readalong_book.json")
        assert manifest["last_read"] == {"chapter": "chapter_001", "unit_id": 2}

        selected = _post_json(base_url + "/api/library/select", {"slug": "book"})
        assert selected["active_book"]["last_read_label"] == "chapter_001, segment 3"

        chapter = _get_json(base_url + "/api/chapter/chapter_001")
        assert chapter["selected_unit_id"] == 2
    finally:
        _stop_server(server, thread)


def test_web_interface_exposes_tts_loading_overlay_and_selection_outline(tmp_path):
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        response = urllib.request.urlopen(base_url, timeout=20).read().decode("utf-8")

        assert 'id="tts-loading-overlay"' in response
        assert "TTS stack loading" in response
        assert ".unit.selected" in response
        assert "outline: 1px dashed" in response
        assert "showTtsLoading(true)" in response
        assert "showTtsLoading(false)" in response
    finally:
        _stop_server(server, thread)


def test_chapter_payload_does_not_build_units_or_tts_on_reader_open(tmp_path):
    paths = _write_demo_book(tmp_path)
    calls = []

    def forbidden_pipeline_factory(config, needs_llm, fake_tts):
        calls.append((config.tts_backend, needs_llm, fake_tts))
        raise AssertionError("opening a chapter must not build a pipeline")

    server = create_server(
        book_root=paths.root,
        host="127.0.0.1",
        port=0,
        fake_tts=False,
        pipeline_factory=forbidden_pipeline_factory,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        state = _get_json(base_url + "/api/state")
        chapter = _get_json(base_url + "/api/chapter/chapter_001")

        assert state["chapters"][0]["chapter"] == "chapter_001"
        assert chapter["text"] == paths.chapter_text("chapter_001").read_text(encoding="utf-8")
        assert chapter["units"] == []
        assert chapter["units_ready"] is False
        assert chapter["annotation_ready"] is True
        assert "Generate Voices" in chapter["message"]
        assert calls == []
    finally:
        _stop_server(server, thread)


def test_web_api_saves_read_along_settings(tmp_path):
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        saved = _post_json(
            base_url + "/api/settings",
            {
                "playback_speed": "1.4",
                "generation_mode": "fast",
                "buffer_limit": "3",
                "narrator_voice_type": "female",
            },
        )

        assert saved["ok"] is True
        assert saved["settings"] == {
            "playback_speed": 1.4,
            "generation_mode": "fast",
            "buffer_limit": 3,
            "target_buffer_seconds": 20.0,
            "start_buffer_seconds": 20.0,
            "max_buffer_seconds": 40.0,
            "max_buffer_units": 32,
            "narrator_voice_type": "female",
        }
        assert _get_json(base_url + "/api/state")["settings"] == saved["settings"]
    finally:
        _stop_server(server, thread)


def test_web_api_returns_json_errors(tmp_path):
    server, thread, base_url = _start_test_server(tmp_path)
    try:
        error = _post_json(base_url + "/api/session/start", {"chapter": ""}, expect_status=400)

        assert error["ok"] is False
        assert "chapter" in error["error"].lower()
    finally:
        _stop_server(server, thread)


def _start_test_server(tmp_path):
    _write_demo_book(tmp_path, ready_for_tts=True)
    server = create_server(book_root=tmp_path / "book", host="127.0.0.1", port=0, fake_tts=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _start_test_server_for_root(root: Path):
    server = create_server(launch_root=root, host="127.0.0.1", port=0, fake_tts=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _write_demo_book(
    tmp_path,
    name: str = "book",
    title: str = "Demo",
    ready_for_tts: bool = False,
) -> BookPaths:
    paths = BookPaths(tmp_path / name)
    paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right." Then she left.', encoding="utf-8")
    write_json_atomic(
        paths.root / "toc.json",
        {
            "chapters": [
                {
                    "index": 1,
                    "chapter": "chapter_001",
                    "title": "Chapter One",
                    "source": "chapter_001.txt",
                }
            ]
        },
    )
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": title, "slug": name},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_identity": {"seed": 1, "differentiators": []},
                "voice_config_path": "voices/narrator.qvp",
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "voice_config_path": "voices/leigh_adult.qvp",
                    "identity_profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["direct"],
                    },
                    "voice_identity": {"seed": 2, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                }
            },
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    (paths.root / "voices").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "narrator.qvp").write_bytes(b"voice")
    (paths.root / "voices" / "leigh_adult.qvp").write_bytes(b"voice")
    if ready_for_tts:
        (paths.root / "voices" / "_samples").mkdir(parents=True, exist_ok=True)
        (paths.root / "voices" / "_samples" / "leigh_adult.wav").write_bytes(b"sample")
        units = [
            {
                "chapter": "chapter_001",
                "unit_id": 0,
                "text": "Leigh said,",
                "source_start": 0,
                "source_end": 11,
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "voice_config_path": "voices/narrator.qvp",
                "quote_id": None,
                "sentence_idx": 0,
                "character": None,
                "voice_variant": None,
            },
            {
                "chapter": "chapter_001",
                "unit_id": 1,
                "text": '"Right."',
                "source_start": 12,
                "source_end": 20,
                "role": "Leigh",
                "role_id": "leigh_adult",
                "type": "dialogue",
                "voice_config_path": "voices/leigh_adult.qvp",
                "quote_id": "q0",
                "sentence_idx": 1,
                "character": "Leigh",
                "voice_variant": None,
            },
            {
                "chapter": "chapter_001",
                "unit_id": 2,
                "text": "Then she left.",
                "source_start": 21,
                "source_end": 35,
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "voice_config_path": "voices/narrator.qvp",
                "quote_id": None,
                "sentence_idx": 2,
                "character": None,
                "voice_variant": None,
            },
        ]
        write_json_atomic(paths.read_along_units("chapter_001"), {"chapter": "chapter_001", "units": units})
    if paths.read_along_units("chapter_001").exists():
        units = read_json(paths.read_along_units("chapter_001"))
        assert units
    return paths


class _FakeExtractor:
    def extract(self, epub_path, paths: BookPaths) -> EpubExtractResult:
        paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text("chapter_001").write_text('Hello. "Welcome."', encoding="utf-8")
        return EpubExtractResult(chapters=["chapter_001"], sources=[str(epub_path)])


class _TwoChapterExtractor:
    def extract(self, epub_path, paths: BookPaths) -> EpubExtractResult:
        paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text("chapter_001").write_text('One. "Welcome."', encoding="utf-8")
        paths.chapter_text("chapter_002").write_text('Two. "Stop."', encoding="utf-8")
        return EpubExtractResult(chapters=["chapter_001", "chapter_002"], sources=[str(epub_path), str(epub_path)])


def _fake_lifecycle_pipeline_factory(config, needs_llm, fake_tts):
    return _FakeLifecyclePipeline(config)


class _FakeLifecyclePipeline:
    def __init__(self, config) -> None:
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.segmenter = SentenceSegmenter()
        self.tts_adapter = FakeTtsAdapter()

    def segment_chapter(self, chapter: str):
        return self.segmenter.segment_chapter(self.paths, chapter)

    def build_global_registry(self, book_title=None, book_slug=None) -> int:
        self.registry.initialize_if_missing(
            book_title=book_title or "Lifecycle Book",
            book_slug=book_slug or self.paths.root.name,
        )
        registry = self.registry.load()
        registry.setdefault("characters", {})["leigh_adult"] = {
            "role_id": "leigh_adult",
            "profile_id": "leigh_adult",
            "person_id": "leigh",
            "display_name": "Leigh",
            "age_stage": "adult",
            "aliases": [],
            "voice_config_path": None,
            "identity_profile": {
                "age_stage": "adult",
                "gender": "female",
                "personality": ["direct"],
            },
            "voice_identity": {"seed": 2, "differentiators": []},
            "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
        }
        self.registry.save(registry)
        return 1

    def annotate_chapter(self, chapter: str, lock_registry: bool = True):
        payload = {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[0, 0]]}
        write_json_atomic(self.paths.annotation(chapter), payload)
        return payload

    def build_read_along_units(self, chapter: str):
        text = self.paths.chapter_text(chapter).read_text(encoding="utf-8")
        registry = self.registry.load()
        narrator_voice = registry.get("narrator", {}).get("voice_config_path")
        leigh_voice = registry.get("characters", {}).get("leigh_adult", {}).get("voice_config_path")
        quote_start = text.index('"')
        quote_end = text.rindex('"') + 1
        units = [
            {
                "chapter": chapter,
                "unit_id": 0,
                "text": text[:quote_start].strip(),
                "source_start": 0,
                "source_end": quote_start,
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "voice_config_path": narrator_voice,
                "quote_id": None,
                "sentence_idx": 0,
                "character": None,
                "voice_variant": None,
            },
            {
                "chapter": chapter,
                "unit_id": 1,
                "text": text[quote_start:quote_end],
                "source_start": quote_start,
                "source_end": quote_end,
                "role": "Leigh",
                "role_id": "leigh_adult",
                "type": "dialogue",
                "voice_config_path": leigh_voice,
                "quote_id": "q0",
                "sentence_idx": 1,
                "character": "Leigh",
                "voice_variant": None,
            },
        ]
        write_json_atomic(self.paths.read_along_units(chapter), {"chapter": chapter, "units": units})
        return units

    def prepare_voices_for_annotation(self, annotation, chapter=None, include_narrator=True):
        registry = self.registry.load()
        if include_narrator:
            narrator = registry.setdefault("narrator", {"role_id": "narrator", "display_name": "Narrator"})
            narrator["voice_config_path"] = "voices/narrator.qvp"
        characters = registry.setdefault("characters", {})
        if "leigh_adult" in characters:
            characters["leigh_adult"]["voice_config_path"] = "voices/leigh_adult.qvp"
        self.registry.save(registry)
        (self.paths.root / "voices").mkdir(parents=True, exist_ok=True)
        if include_narrator:
            (self.paths.root / "voices" / "narrator.qvp").write_bytes(b"voice")
        (self.paths.root / "voices" / "leigh_adult.qvp").write_bytes(b"voice")
        return None


class _FailingSecondChapterPipeline(_FakeLifecyclePipeline):
    def annotate_chapter(self, chapter: str, lock_registry: bool = True):
        if chapter == "chapter_002":
            raise RuntimeError("model timed out")
        return super().annotate_chapter(chapter, lock_registry=lock_registry)


def _failing_second_chapter_pipeline_factory(config, needs_llm, fake_tts):
    return _FailingSecondChapterPipeline(config)


class _FailOnceSecondChapterPipelineFactory:
    def __init__(self) -> None:
        self.failures_remaining = 1
        self.annotate_calls = []
        self.unit_calls = []

    def __call__(self, config, needs_llm, fake_tts):
        return _FailOnceSecondChapterPipeline(config, self)


class _FailOnceSecondChapterPipeline(_FakeLifecyclePipeline):
    def __init__(self, config, factory: _FailOnceSecondChapterPipelineFactory) -> None:
        super().__init__(config)
        self.factory = factory

    def annotate_chapter(self, chapter: str, lock_registry: bool = True):
        self.factory.annotate_calls.append(chapter)
        if chapter == "chapter_002" and self.factory.failures_remaining > 0:
            self.factory.failures_remaining -= 1
            raise RuntimeError("model timed out")
        return super().annotate_chapter(chapter, lock_registry=lock_registry)

    def build_read_along_units(self, chapter: str):
        self.factory.unit_calls.append(chapter)
        return super().build_read_along_units(chapter)


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def _get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def _get_json(url: str) -> dict:
    return json.loads(_get_text(url))


def _wait_for_job(base_url: str, job_id: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    last = {}
    while time.time() < deadline:
        last = _get_json(base_url + "/api/library/job-status?job_id=" + job_id)
        if last["job"]["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")


def _post_json(url: str, payload: dict, expect_status: int = 200) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.status == expect_status
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        assert exc.code == expect_status
        return json.loads(exc.read().decode("utf-8"))


def _post_multipart(
    url: str,
    fields: dict,
    files: dict,
    expect_status: int = 200,
) -> dict:
    boundary = "----readalong-test-boundary"
    chunks = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, content_type, content) in files.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.status == expect_status
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        assert exc.code == expect_status
        return json.loads(exc.read().decode("utf-8"))
