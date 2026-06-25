from __future__ import annotations

import io
import json
import zipfile

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import voice_profile_hash
from ebook_tts_pipeline.ui.book_package import build_readalong_package, import_readalong_package


def test_build_readalong_package_includes_portable_assets_and_excludes_runtime_artifacts(tmp_path):
    paths = _write_portable_book(tmp_path / "share-me")
    archive = build_readalong_package(paths.root)

    names = _zip_names(archive)

    assert "readalong_package.json" in names
    assert "readalong_book.json" in names
    assert "registry.json" in names
    assert "toc.json" in names
    assert "chapters/chapter_001.txt" in names
    assert "sentence_segments/chapter_001.sentences.json" in names
    assert "annotations/chapter_001.annotation.json" in names
    assert "read_along/chapter_001.units.json" in names
    assert "read_along/settings.json" in names
    assert "read_along/narrator_profile.json" in names
    assert "temp_registries/chapter_001.temp_registry.json" in names
    assert "voices/leigh_adult.qvp" in names
    assert "voices/_samples/leigh_adult.wav" in names

    assert "_source/original.epub" not in names
    assert "logs/readalong_web_errors.jsonl" not in names
    assert "read_along/annotation_progress.json" not in names
    assert "read_along_sessions/session/timings.jsonl" not in names
    assert "voices/narrator.qvp" not in names
    assert "voices/_narrator/hash/narrator.qvp" not in names
    assert "voices/_temp/chapter_001/local.qvp" not in names

    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        manifest = json.loads(package.read("readalong_book.json").decode("utf-8"))
        package_manifest = json.loads(package.read("readalong_package.json").decode("utf-8"))

    assert manifest["slug"] == "share-me"
    assert "last_read" not in manifest
    assert manifest["source_epub_path"] == ""
    assert package_manifest["schema"] == "readalong_package_v1"
    assert package_manifest["source_slug"] == "share-me"


def test_import_readalong_package_creates_safe_ready_book_without_personal_state(tmp_path):
    source_paths = _write_portable_book(tmp_path / "source-book")
    archive = build_readalong_package(source_paths.root)
    library_root = tmp_path / "library"

    imported = import_readalong_package(library_root, archive, requested_slug="Shared Copy")

    assert imported == library_root / "shared-copy"
    assert (imported / "chapters" / "chapter_001.txt").read_text(encoding="utf-8")
    assert (imported / "read_along" / "chapter_001.units.json").exists()
    assert (imported / "voices" / "leigh_adult.qvp").read_bytes() == b"global voice"
    assert (imported / "voices" / "_samples" / "leigh_adult.wav").read_bytes() == b"sample voice"
    assert not (imported / "_source").exists()
    assert not (imported / "logs").exists()
    assert not (imported / "read_along_sessions").exists()
    assert not (imported / "voices" / "_temp").exists()
    assert not (imported / "voices" / "_narrator").exists()

    manifest = read_json(imported / "readalong_book.json")
    assert manifest["slug"] == "shared-copy"
    assert manifest["title"] == "Portable Book"
    assert manifest["source_epub_path"] == ""
    assert "last_read" not in manifest
    assert manifest["stages"]["initialized"] is True
    assert manifest["stages"]["global_registry"] is True
    assert manifest["stages"]["annotated"] is True
    assert manifest["stages"]["voices_ready"] is True


def _zip_names(archive: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        return set(package.namelist())


def _write_portable_book(root) -> BookPaths:
    paths = BookPaths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    write_json_atomic(
        paths.root / "readalong_book.json",
        {
            "schema": "readalong_book_v1",
            "title": "Portable Book",
            "author": "Karin Slaughter",
            "slug": paths.root.name,
            "source_epub_path": "_source/original.epub",
            "original_filename": "Portable.epub",
            "stages": {
                "source_added": True,
                "initialized": True,
                "global_registry": True,
                "annotating": False,
                "annotated": True,
                "registry_reviewed": True,
                "voices_ready": True,
            },
            "last_read": {"chapter": "chapter_001", "unit_id": 1},
        },
    )
    write_json_atomic(paths.root / "toc.json", {"chapters": [{"chapter": "chapter_001"}]})
    leigh = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "voice_config_path": "voices/leigh_adult.qvp",
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
        "voice_identity": {"seed": 5, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
    }
    leigh["voice_config_hash"] = voice_profile_hash(leigh)
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Portable Book", "slug": paths.root.name},
            "narrator": {"role_id": "narrator", "voice_config_path": "voices/narrator.qvp"},
            "characters": {"leigh_adult": leigh},
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.annotation("chapter_001"), {"schema": "quote_attribution_v1", "quotes": [[1, 0]]})
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.sentence_artifact("chapter_001"), {"sentences": []})
    write_json_atomic(
        paths.read_along_units("chapter_001"),
        {
            "chapter": "chapter_001",
            "units": [
                {
                    "chapter": "chapter_001",
                    "unit_id": 0,
                    "text": "Leigh said.",
                    "source_start": 0,
                    "source_end": 11,
                    "role": "Leigh",
                    "role_id": "leigh_adult",
                    "type": "dialogue",
                    "voice_config_path": "voices/leigh_adult.qvp",
                    "quote_id": "q1",
                    "sentence_idx": 0,
                    "character": "Leigh",
                    "voice_variant": None,
                }
            ],
        },
    )
    write_json_atomic(paths.root / "read_along" / "settings.json", {"generation_mode": "balanced"})
    write_json_atomic(paths.root / "read_along" / "narrator_profile.json", {"role_id": "narrator"})
    paths.chapter_temp_registry("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.chapter_temp_registry("chapter_001"), {"speakers": {}})

    (paths.root / "_source").mkdir(parents=True, exist_ok=True)
    (paths.root / "_source" / "original.epub").write_bytes(b"epub")
    (paths.root / "logs").mkdir(parents=True, exist_ok=True)
    (paths.root / "logs" / "readalong_web_errors.jsonl").write_text("debug", encoding="utf-8")
    (paths.root / "read_along" / "annotation_progress.json").write_text("{}", encoding="utf-8")
    (paths.root / "read_along_sessions" / "session").mkdir(parents=True, exist_ok=True)
    (paths.root / "read_along_sessions" / "session" / "timings.jsonl").write_text("{}", encoding="utf-8")
    (paths.root / "voices" / "_samples").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "_narrator" / "hash").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "_temp" / "chapter_001").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "narrator.qvp").write_bytes(b"narrator artifact")
    (paths.root / "voices" / "leigh_adult.qvp").write_bytes(b"global voice")
    (paths.root / "voices" / "_samples" / "leigh_adult.wav").write_bytes(b"sample voice")
    (paths.root / "voices" / "_narrator" / "hash" / "narrator.qvp").write_bytes(b"runtime narrator")
    (paths.root / "voices" / "_temp" / "chapter_001" / "local.qvp").write_bytes(b"runtime local")
    return paths
