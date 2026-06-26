from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from ebook_tts_pipeline.json_io import read_json, write_json_atomic


BOOK_MANIFEST = "readalong_book.json"
PACKAGE_MANIFEST = "readalong_package.json"
PACKAGE_SCHEMA = "readalong_package_v1"

_ROOT_FILES = {"registry.json", "toc.json"}
_ROOT_EXACT = {PACKAGE_MANIFEST, BOOK_MANIFEST, *_ROOT_FILES}
_ALLOWED_PREFIXES = ("chapters/", "sentence_segments/", "annotations/", "temp_registries/")
_READ_ALONG_EXACT = {"read_along/settings.json", "read_along/narrator_profile.json"}
_AUDIOBOOK_EXACT = {"audiobook/settings.json", "audiobook/narrator_profile.json"}
_EXCLUDED_PREFIXES = (
    "_source/",
    "logs/",
    "read_along_sessions/",
    "voices/_temp/",
    "voices/_narrator/",
)


def build_readalong_package(book_root: str | Path) -> bytes:
    root = Path(book_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Book folder not found: {root}")
    manifest = _read_json_if_exists(root / BOOK_MANIFEST)
    registry = _read_json_if_exists(root / "registry.json")
    title = str(manifest.get("title") or _book_title_from_registry(registry) or root.name)
    author = str(manifest.get("author") or "")
    slug = str(manifest.get("slug") or root.name)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            PACKAGE_MANIFEST,
            _json_bytes(
                {
                    "schema": PACKAGE_SCHEMA,
                    "source_slug": slug,
                    "title": title,
                    "author": author,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "source_epub_included": False,
                }
            ),
        )
        package.writestr(
            BOOK_MANIFEST,
            _json_bytes(_portable_book_manifest(manifest, slug=slug, title=title, author=author)),
        )
        for relative in _portable_relative_paths(root, registry):
            package.write(root / relative, relative.as_posix())
    return buffer.getvalue()


def import_readalong_package(
    library_root: str | Path,
    archive_bytes: bytes,
    requested_slug: str = "",
) -> Path:
    root = Path(library_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = _read_archive_entries(archive_bytes)
    manifest = _json_entry(entries, BOOK_MANIFEST)
    package_manifest = _json_entry(entries, PACKAGE_MANIFEST)
    if package_manifest and package_manifest.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("Unsupported ReadAlong package schema.")
    if not manifest:
        raise ValueError("ReadAlong package is missing readalong_book.json.")

    title = str(manifest.get("title") or package_manifest.get("title") or "Imported Book")
    author = str(manifest.get("author") or package_manifest.get("author") or "")
    source_slug = str(manifest.get("slug") or package_manifest.get("source_slug") or title)
    slug = _unique_slug(root, _safe_slug(requested_slug or source_slug or title))
    target = root / slug
    target.mkdir(parents=True)
    try:
        for relative, content in entries.items():
            if relative in {PACKAGE_MANIFEST, BOOK_MANIFEST}:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        write_json_atomic(
            target / BOOK_MANIFEST,
            _portable_book_manifest(manifest, slug=slug, title=title, author=author),
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def _portable_relative_paths(root: Path, registry: Dict[str, Any]) -> List[Path]:
    paths: Set[Path] = set()
    for name in _ROOT_FILES:
        if (root / name).exists():
            paths.add(Path(name))
    for dirname in ("chapters", "sentence_segments", "annotations", "temp_registries"):
        paths.update(_recursive_files(root, dirname))
    read_along = root / "read_along"
    if read_along.exists():
        for file in sorted(read_along.glob("*.units.json")):
            paths.add(file.relative_to(root))
        for name in ("settings.json", "narrator_profile.json"):
            if (read_along / name).exists():
                paths.add(Path("read_along") / name)
    audiobook = root / "audiobook"
    if audiobook.exists():
        for name in ("settings.json", "narrator_profile.json"):
            if (audiobook / name).exists():
                paths.add(Path("audiobook") / name)
    paths.update(_registry_voice_paths(root, registry))
    return sorted(paths, key=lambda item: item.as_posix())


def _recursive_files(root: Path, dirname: str) -> Iterable[Path]:
    folder = root / dirname
    if not folder.exists():
        return []
    return sorted(file.relative_to(root) for file in folder.rglob("*") if file.is_file())


def _registry_voice_paths(root: Path, registry: Dict[str, Any]) -> Set[Path]:
    paths: Set[Path] = set()
    characters = registry.get("characters", {}) if isinstance(registry.get("characters"), dict) else {}
    for record in characters.values():
        if not isinstance(record, dict):
            continue
        relative = _clean_relative_path(str(record.get("voice_config_path") or ""))
        if relative and _is_allowed_voice_qvp(relative) and (root / relative).exists():
            paths.add(Path(relative))
    sample_root = root / "voices" / "_samples"
    if sample_root.exists():
        for sample in sorted(sample_root.glob("*.wav")):
            paths.add(sample.relative_to(root))
    return paths


def _read_archive_entries(archive_bytes: bytes) -> Dict[Path, bytes]:
    entries: Dict[Path, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as package:
        for info in package.infolist():
            if info.is_dir():
                continue
            relative = _validated_archive_path(info.filename)
            if not _is_allowed_package_path(relative):
                raise ValueError(f"ReadAlong package contains unsupported path: {relative.as_posix()}")
            entries[relative] = package.read(info)
    _require_import_assets(entries)
    return entries


def _validated_archive_path(name: str) -> Path:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"ReadAlong package contains unsafe path: {name}")
    path = Path(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"ReadAlong package contains unsafe path: {name}")
    return path


def _is_allowed_package_path(relative: Path) -> bool:
    value = relative.as_posix()
    if value.startswith(_EXCLUDED_PREFIXES):
        return False
    if value in _ROOT_EXACT:
        return True
    if value.startswith(_ALLOWED_PREFIXES):
        return True
    if value in _READ_ALONG_EXACT:
        return True
    if value in _AUDIOBOOK_EXACT:
        return True
    if value.startswith("read_along/") and value.endswith(".units.json"):
        return True
    if _is_allowed_voice_qvp(value):
        return True
    if value.startswith("voices/_samples/") and value.endswith(".wav") and len(relative.parts) == 3:
        return True
    return False


def _is_allowed_voice_qvp(value: str) -> bool:
    path = Path(value)
    return (
        len(path.parts) == 2
        and path.parts[0] == "voices"
        and path.suffix == ".qvp"
        and path.name != "narrator.qvp"
    )


def _require_import_assets(entries: Dict[Path, bytes]) -> None:
    required = {Path(BOOK_MANIFEST), Path("registry.json"), Path("toc.json")}
    missing = [item.as_posix() for item in required if item not in entries]
    if missing:
        raise ValueError(f"ReadAlong package is missing required file(s): {', '.join(sorted(missing))}")
    if not any(path.as_posix().startswith("chapters/") and path.suffix == ".txt" for path in entries):
        raise ValueError("ReadAlong package must include at least one chapter text file.")
    if not any(path.as_posix().startswith("read_along/") and path.name.endswith(".units.json") for path in entries):
        raise ValueError("ReadAlong package must include read-along units.")


def _portable_book_manifest(manifest: Dict[str, Any], slug: str, title: str, author: str) -> Dict[str, Any]:
    payload = dict(manifest)
    payload["schema"] = "readalong_book_v1"
    payload["title"] = title
    payload["author"] = author
    payload["slug"] = slug
    payload["source_epub_path"] = ""
    payload["source_included"] = False
    payload.pop("last_read", None)
    stages = dict(payload.get("stages", {})) if isinstance(payload.get("stages"), dict) else {}
    stages.update(
        {
            "source_added": False,
            "initialized": True,
            "global_registry": True,
            "annotating": False,
            "annotated": True,
            "registry_reviewed": True,
            "voices_ready": True,
        }
    )
    payload["stages"] = stages
    return payload


def _json_entry(entries: Dict[Path, bytes], name: str) -> Dict[str, Any]:
    content = entries.get(Path(name))
    if content is None:
        return {}
    payload = json.loads(content.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_bytes(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _book_title_from_registry(registry: Dict[str, Any]) -> str:
    book = registry.get("book") if isinstance(registry.get("book"), dict) else {}
    return str(book.get("title") or "")


def _clean_relative_path(value: str) -> str:
    cleaned = value.replace("\\", "/").strip().lstrip("/")
    if not cleaned or ".." in Path(cleaned).parts:
        return ""
    return cleaned


def _unique_slug(library_root: Path, base: str) -> str:
    slug = base or "imported-book"
    candidate = slug
    index = 2
    while (library_root / candidate).exists():
        candidate = f"{slug}-{index}"
        index += 1
    return candidate


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "imported-book"
