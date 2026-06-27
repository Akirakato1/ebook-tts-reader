from __future__ import annotations

import argparse
import cgi
import io
import json
import re
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.read_along.session import BufferedAudio, ReadAlongSession
from ebook_tts_pipeline.registry import voice_profile_hash
from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.ui.book_package import build_readalong_package, import_readalong_package
from ebook_tts_pipeline.ui.controller import (
    ChapterExtractor,
    PipelineFactory,
    PrototypeUiController,
    _annotation_matches_chapter_quotes,
)
from ebook_tts_pipeline.ui.errors import write_readalong_error_event


BOOK_MANIFEST = "readalong_book.json"
SOURCE_EPUB_PATH = Path("_source") / "original.epub"


@dataclass(frozen=True)
class LibraryBookSummary:
    title: str
    author: str
    slug: str
    book_root: Path
    chapter_count: int
    annotation_count: int
    read_along_unit_count: int
    voice_count: int
    voice_total: int
    audio_count: int
    has_registry: bool
    status_key: str
    status_label: str
    status_detail: str
    action_key: str
    action_label: str
    open_enabled: bool
    resume_annotation_enabled: bool
    resume_annotation_label: str
    last_read: Dict[str, Any]
    last_read_label: str

    def to_payload(self, active: bool = False) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "slug": self.slug,
            "book_root": str(self.book_root),
            "chapter_count": self.chapter_count,
            "annotation_count": self.annotation_count,
            "read_along_unit_count": self.read_along_unit_count,
            "voice_count": self.voice_count,
            "voice_total": self.voice_total,
            "audio_count": self.audio_count,
            "has_registry": self.has_registry,
            "status_key": self.status_key,
            "status_label": self.status_label,
            "status_detail": self.status_detail,
            "action_key": self.action_key,
            "action_label": self.action_label,
            "open_enabled": self.open_enabled,
            "resume_annotation_enabled": self.resume_annotation_enabled,
            "resume_annotation_label": self.resume_annotation_label,
            "last_read": self.last_read,
            "last_read_label": self.last_read_label,
            "active": active,
        }


@dataclass(frozen=True)
class LaunchSelection:
    library_root: Path
    active_book_root: Optional[Path]


@dataclass(frozen=True)
class UploadedBook:
    filename: str
    content: bytes
    title: str
    author: str
    slug: str


@dataclass(frozen=True)
class UploadedReadAlongPackage:
    filename: str
    content: bytes
    slug: str


@dataclass
class LibraryJob:
    job_id: str
    slug: str
    action_key: str
    status: str = "queued"
    current_chapter: str = ""
    current_item: str = ""
    completed: int = 0
    total: int = 0
    failed_chapter: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    chapters: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    force: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "slug": self.slug,
            "action_key": self.action_key,
            "status": self.status,
            "current_chapter": self.current_chapter,
            "current_item": self.current_item,
            "completed": self.completed,
            "total": self.total,
            "failed_chapter": self.failed_chapter,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "chapters": list(self.chapters),
        }


@dataclass
class ReadAlongWebState:
    library_root: Path
    fake_tts: bool = False
    extractor: Optional[ChapterExtractor] = None
    pipeline_factory: Optional[PipelineFactory] = None
    controller: Optional[PrototypeUiController] = None
    session: Optional[ReadAlongSession] = None
    session_chapter: str = ""

    def __post_init__(self) -> None:
        self.library_root = Path(self.library_root).resolve()
        self.lock = threading.RLock()
        self.session_fill_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.jobs: Dict[str, LibraryJob] = {}
        self.jobs_by_slug: Dict[str, str] = {}
        self.session_start_progress: Dict[str, Any] = {
            "status": "idle",
            "stage": "idle",
            "message": "Idle.",
            "updated_at": time.time(),
        }

    def library_payload(self) -> Dict[str, Any]:
        with self.lock:
            active_root = self.controller.book_root if self.controller is not None else None
            books = [
                summary.to_payload(active=_same_path(summary.book_root, active_root))
                for summary in discover_books(self.library_root)
            ]
            active_book = self._active_book_payload()
            return {
                "ok": True,
                "mode": "book" if active_book is not None else "library",
                "library_root": str(self.library_root),
                "active_book": active_book,
                "books": books,
            }

    def state_payload(self) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            return {
                "ok": True,
                "active_book": self._active_book_payload(),
                "chapters": [
                    {
                        "chapter": row.chapter,
                        "index": row.index,
                        "title": row.title,
                        "stage": row.stage.value,
                    }
                    for row in controller.chapter_rows()
                ],
                "settings": controller.read_along_settings(),
                "session_active": self.session is not None,
                "session_chapter": self.session_chapter,
            }

    def session_start_progress_payload(self) -> Dict[str, Any]:
        with self.progress_lock:
            return {"ok": True, **dict(self.session_start_progress)}

    def log_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        exc: Optional[BaseException] = None,
        book_root: Optional[Path] = None,
    ) -> Optional[Path]:
        try:
            with self.lock:
                root = Path(book_root) if book_root is not None else (
                    self.controller.book_root if self.controller is not None else self.library_root
                )
            log_path = write_readalong_error_event(root, event_type, details or {}, exc=exc)
            log_runtime_step("web_readalong_event_logged", event_type=event_type, log_path=log_path)
            return log_path
        except Exception as log_exc:
            print(f"[ebook-tts] failed_to_write_readalong_log event_type={event_type} error={log_exc}", flush=True)
            return None

    def chapter_payload(self, chapter: str) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            units_path = controller.paths.read_along_units(chapter)
            units_ready = units_path.exists()
            annotation_ready = controller.paths.annotation(chapter).exists()
            units = list(read_json(units_path).get("units", [])) if units_ready else []
            selected_unit_id = _selected_last_read_unit_id(controller.book_root, chapter, units)
            message = (
                "Ready for read-along."
                if units
                else (
                    "Generate Voices before starting read-along for this chapter."
                    if annotation_ready
                    else "Process Book before starting read-along for this chapter."
                )
            )
            return {
                "ok": True,
                "chapter": chapter,
                "text": controller.chapter_text(chapter),
                "units": units,
                "units_ready": bool(units),
                "annotation_ready": annotation_ready,
                "selected_unit_id": selected_unit_id,
                "message": message,
            }

    def select_book(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            normalized = str(slug).strip()
            if not normalized:
                raise ValueError("slug is required")
            for summary in discover_books(self.library_root):
                if summary.slug == normalized:
                    if not summary.open_enabled:
                        raise ValueError(f"{summary.action_label} is required before opening this book.")
                    self.end_session()
                    self.controller = self._make_controller(summary.book_root)
                    return {"ok": True, "active_book": summary.to_payload(active=True)}
            raise ValueError(f"Book not found: {slug}")

    def add_book(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            epub_text = str(payload.get("epub_path", "")).strip()
            if not epub_text:
                raise ValueError("epub_path is required")
            epub_path = Path(epub_text)
            if not epub_path.exists():
                raise FileNotFoundError(f"EPUB not found: {epub_path}")
            title = str(payload.get("title", "")).strip() or epub_path.stem or "Untitled Book"
            author = str(payload.get("author", "")).strip()
            slug = _safe_slug(str(payload.get("slug", "")).strip() or title)
            target_root = self._book_root_for_slug(slug)
            if target_root.exists() and any(target_root.iterdir()):
                raise ValueError(f"Book folder already exists and is not empty: {target_root}")
            source_path = target_root / SOURCE_EPUB_PATH
            source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(epub_path, source_path)
            _write_book_manifest(target_root, title=title, author=author, slug=slug, original_filename=epub_path.name)
            summary = summarize_book(target_root)
            return {"ok": True, "book": summary.to_payload(), "active_book": None, "library": self.library_payload()}

    def add_uploaded_book(self, upload: UploadedBook) -> Dict[str, Any]:
        with self.lock:
            if not upload.content:
                raise ValueError("epub upload is required")
            filename_stem = Path(upload.filename or "book.epub").stem
            title = upload.title.strip() or filename_stem or "Untitled Book"
            author = upload.author.strip()
            slug = _safe_slug(upload.slug.strip() or title)
            target_root = self._book_root_for_slug(slug)
            if target_root.exists() and any(target_root.iterdir()):
                raise ValueError(f"Book folder already exists and is not empty: {target_root}")
            source_dir = target_root / "_source"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_path = source_dir / "original.epub"
            source_path.write_bytes(upload.content)
            _write_book_manifest(target_root, title=title, author=author, slug=slug, original_filename=upload.filename)
            summary = summarize_book(target_root)
            return {"ok": True, "book": summary.to_payload(), "active_book": None, "library": self.library_payload()}

    def initialize_book(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            manifest = _read_book_manifest(summary.book_root)
            source_path = summary.book_root / str(manifest.get("source_epub_path", SOURCE_EPUB_PATH.as_posix()))
            if not source_path.exists():
                raise FileNotFoundError(f"Source EPUB not found: {source_path}")
            controller = self._make_controller(summary.book_root)
            controller.load_epub(
                source_path,
                title=str(manifest.get("title", summary.title)),
                author=str(manifest.get("author", summary.author)),
                slug=str(manifest.get("slug", summary.slug)),
            )
            _update_book_stage(summary.book_root, initialized=True)
            summary = summarize_book(summary.book_root)
            return {"ok": True, "book": summary.to_payload(), "library": self.library_payload()}

    def build_book_registry(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            if summary.status_key == "fresh_added":
                raise ValueError("Initialize Book is required before building the registry.")
            controller = self._make_controller(summary.book_root)
            count = controller.build_global_registry()
            _update_book_stage(summary.book_root, global_registry=True)
            summary = summarize_book(summary.book_root)
            return {"ok": True, "book": summary.to_payload(), "registry_characters": count, "library": self.library_payload()}

    def annotate_book(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            if summary.status_key == "fresh_added":
                raise ValueError("Initialize Book is required before annotation.")
            if summary.status_key == "initialized":
                raise ValueError("Build Registry is required before annotation.")
            existing = self._running_job_for_slug(summary.slug)
            if existing is not None:
                return {
                    "ok": True,
                    "book": summary.to_payload(),
                    "job": existing.to_payload(),
                    "library": self.library_payload(),
                }
            job = LibraryJob(
                job_id=uuid.uuid4().hex,
                slug=summary.slug,
                action_key="annotate",
                total=summary.chapter_count,
            )
            self.jobs[job.job_id] = job
            self.jobs_by_slug[summary.slug] = job.job_id
            _update_book_stage(summary.book_root, annotating=True)
            summary = summarize_book(summary.book_root)
            worker = threading.Thread(target=self._run_annotate_job, args=(job.job_id,), daemon=True)
            worker.start()
            return {
                "ok": True,
                "book": summary.to_payload(),
                "job": job.to_payload(),
                "library": self.library_payload(),
            }

    def _run_annotate_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            summary = self._require_book_summary(job.slug)
            controller = self._make_controller(summary.book_root)
            job.status = "running"

        def progress_callback(event: Dict[str, Any]) -> None:
            with self.lock:
                current = self.jobs[job_id]
                current.total = int(event.get("total") or current.total or 0)
                current.current_chapter = str(event.get("chapter") or current.current_chapter)
                status = str(event.get("status") or "")
                if status == "started":
                    current.completed = max(0, int(event.get("index") or 1) - 1)
                if status == "completed":
                    current.completed = max(current.completed, int(event.get("index") or current.completed))
                if status == "failed":
                    current.failed_chapter = str(event.get("chapter") or "")
                    current.error = str(event.get("error") or "")

        try:
            result = controller.annotate_read_along_book(progress_callback=progress_callback)
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                job.status = "failed"
                if not job.failed_chapter:
                    job.failed_chapter = job.current_chapter
                job.error = str(exc)
                job.finished_at = time.time()
                summary = self._require_book_summary(job.slug)
                _update_book_stage(summary.book_root, annotating=False, annotated=False)
                self.log_event(
                    "library_annotate_job_error",
                    {
                        "job_id": job_id,
                        "slug": job.slug,
                        "current_chapter": job.current_chapter,
                        "failed_chapter": job.failed_chapter,
                    },
                    exc=exc,
                    book_root=summary.book_root,
                )
            return

        with self.lock:
            job = self.jobs[job_id]
            job.status = "completed"
            job.completed = int(result.get("chapters", job.completed))
            job.total = int(result.get("chapters", job.total))
            job.current_chapter = ""
            job.finished_at = time.time()
            summary = self._require_book_summary(job.slug)
            _update_book_stage(summary.book_root, annotating=False, annotated=True)

    def job_status(self, job_id: str) -> Dict[str, Any]:
        with self.lock:
            job = self.jobs.get(str(job_id).strip())
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            summary = self._require_book_summary(job.slug)
            return {
                "ok": True,
                "job": job.to_payload(),
                "book": summary.to_payload(),
                "library": self.library_payload(),
            }

    def _running_job_for_slug(self, slug: str) -> Optional[LibraryJob]:
        job_id = self.jobs_by_slug.get(slug)
        if not job_id:
            return None
        job = self.jobs.get(job_id)
        if job is None or job.status in {"completed", "failed"}:
            return None
        return job

    def registry_review(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            controller = self._make_controller(summary.book_root)
            return {
                "ok": True,
                "book": summary.to_payload(),
                "review": controller.registry_review_payload(),
                "library": self.library_payload(),
            }

    def registry_save_character(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(str(payload.get("slug", "")))
            controller = self._make_controller(summary.book_root)
            role_id = str(payload.get("role_id", ""))
            controller.save_registry_character_form(
                role_id,
                {str(key): str(value) for key, value in dict(payload.get("fields", {})).items()},
            )
            sample = controller.generate_registry_voice_sample(role_id)
            _update_book_stage(summary.book_root, registry_reviewed=False, voices_ready=False)
            summary = summarize_book(summary.book_root)
            return {
                "ok": True,
                "book": summary.to_payload(),
                "sample": sample,
                "review": controller.registry_review_payload(),
                "library": self.library_payload(),
            }

    def registry_save_characters(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(str(payload.get("slug", "")))
            controller = self._make_controller(summary.book_root)
            entries = payload.get("entries", [])
            if not isinstance(entries, list) or not entries:
                raise ValueError("At least one registry character entry is required.")
            changed_role_ids: List[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                role_id = str(entry.get("role_id", "")).strip()
                if not role_id:
                    continue
                changed = controller.save_registry_character_form(
                    role_id,
                    {str(key): str(value) for key, value in dict(entry.get("fields", {})).items()},
                )
                if changed:
                    changed_role_ids.append(role_id)
            sample_count = 0
            if changed_role_ids:
                _update_book_stage(summary.book_root, registry_reviewed=False, voices_ready=False)
            summary = summarize_book(summary.book_root)
            return {
                "ok": True,
                "book": summary.to_payload(),
                "changed_role_ids": changed_role_ids,
                "sample_count": sample_count,
                "review": controller.registry_review_payload(),
                "library": self.library_payload(),
            }

    def registry_confirm(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            if summary.status_key not in {"registry_review", "registry_reviewed", "voices_ready"}:
                raise ValueError("Review Voices is required after annotation.")
            controller = self._make_controller(summary.book_root)
            _update_book_stage(summary.book_root, registry_reviewed=True, voices_ready=False)
            summary = summarize_book(summary.book_root)
            return {
                "ok": True,
                "book": summary.to_payload(),
                "review": controller.registry_review_payload(),
                "library": self.library_payload(),
            }

    def registry_generate_sample(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(str(payload.get("slug", "")))
            controller = self._make_controller(summary.book_root)
            role_id = str(payload.get("role_id", ""))
            log_runtime_step(
                "web_registry_sample_start",
                slug=summary.slug,
                role_id=role_id,
                book_root=summary.book_root,
            )
            sample = dict(controller.generate_registry_voice_sample(role_id))
            sample["sample_url"] = f"{sample['sample_url']}?slug={summary.slug}"
            _update_book_stage(summary.book_root, voices_ready=False)
            summary = summarize_book(summary.book_root)
            log_runtime_step(
                "web_registry_sample_done",
                slug=summary.slug,
                role_id=role_id,
                sample_url=sample.get("sample_url"),
            )
            return {
                "ok": True,
                "book": summary.to_payload(),
                "sample": sample,
                "review": controller.registry_review_payload(),
                "library": self.library_payload(),
            }

    def registry_sample_audio(self, role_id: str, slug: str = "") -> bytes:
        with self.lock:
            if slug:
                book_root = self._require_book_summary(slug).book_root
            else:
                if self.controller is None:
                    raise ValueError("Select or load a book before playing registry samples.")
                book_root = self.controller.book_root
            safe_role = _safe_slug(role_id)
            sample_root = (book_root / "voices" / "_samples").resolve()
            sample_path = (sample_root / f"{safe_role}.wav").resolve()
            if sample_root not in sample_path.parents:
                raise FileNotFoundError("sample path is outside the sample directory")
            return sample_path.read_bytes()

    def prepare_book_voices(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            if summary.status_key not in {"annotated", "registry_review", "registry_reviewed", "voices_ready"}:
                raise ValueError("Annotation is required before generating voices.")
            existing = self._running_job_for_slug(summary.slug)
            if existing is not None:
                return {
                    "ok": True,
                    "book": summary.to_payload(),
                    "job": existing.to_payload(),
                    "library": self.library_payload(),
                }
            job = LibraryJob(
                job_id=uuid.uuid4().hex,
                slug=summary.slug,
                action_key="prepare_voices",
                completed=summary.voice_count,
                total=summary.voice_total,
            )
            self.jobs[job.job_id] = job
            self.jobs_by_slug[summary.slug] = job.job_id
            _update_book_stage(summary.book_root, voices_ready=False)
            summary = summarize_book(summary.book_root)
            log_runtime_step(
                "web_prepare_voices_start",
                slug=summary.slug,
                book_root=summary.book_root,
                status=summary.status_key,
                completed=summary.voice_count,
                total=summary.voice_total,
            )
            worker = threading.Thread(target=self._run_prepare_voices_job, args=(job.job_id,), daemon=True)
            worker.start()
            return {
                "ok": True,
                "book": summary.to_payload(),
                "job": job.to_payload(),
                "library": self.library_payload(),
            }

    def _run_prepare_voices_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            summary = self._require_book_summary(job.slug)
            controller = self._make_controller(summary.book_root)
            job.status = "running"
            job.completed = summary.voice_count
            job.total = summary.voice_total

        def progress_callback(event: Dict[str, Any]) -> None:
            with self.lock:
                current = self.jobs[job_id]
                current.total = int(event.get("total") or current.total or 0)
                current.completed = int(event.get("completed") or current.completed or 0)
                current.current_chapter = str(event.get("chapter") or "")
                current.current_item = str(event.get("display_name") or event.get("role_id") or "")

        try:
            result = controller.prepare_read_along_voices(progress_callback=progress_callback)
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()
                summary = self._require_book_summary(job.slug)
                _update_book_stage(summary.book_root, voices_ready=False)
                self.log_event(
                    "library_prepare_voices_job_error",
                    {
                        "job_id": job_id,
                        "slug": job.slug,
                        "current_chapter": job.current_chapter,
                        "current_item": job.current_item,
                    },
                    exc=exc,
                    book_root=summary.book_root,
                )
            return

        with self.lock:
            job = self.jobs[job_id]
            job.status = "completed"
            job.completed = int(result.get("voice_count", job.completed))
            job.total = int(result.get("voice_total", job.total))
            job.current_chapter = ""
            job.current_item = ""
            job.finished_at = time.time()
            summary = self._require_book_summary(job.slug)
            _update_book_stage(summary.book_root, voices_ready=bool(result["voices_ready"]))
            log_runtime_step(
                "web_prepare_voices_done",
                slug=summary.slug,
                voices_ready=result.get("voices_ready"),
                prepared_chapters=result.get("prepared_chapters"),
                sample_count=result.get("sample_count"),
                voice_count=result.get("voice_count"),
                voice_total=result.get("voice_total"),
                missing_voice_paths=len(result.get("missing_voice_paths") or []),
            )

    def delete_book(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            library_root = self.library_root.resolve()
            target = summary.book_root.resolve()
            try:
                relative = target.relative_to(library_root)
            except ValueError as exc:
                raise ValueError("Book folder must stay inside the library root.") from exc
            if len(relative.parts) != 1:
                raise ValueError("Book folder must be a direct child of the library root.")
            if self.controller is not None and _same_path(self.controller.book_root, summary.book_root):
                self.end_session()
                self.controller = None
            shutil.rmtree(target)
            return {
                "ok": True,
                "deleted_slug": summary.slug,
                "active_book": self._active_book_payload(),
                "library": self.library_payload(),
            }

    def export_book_package(self, slug: str) -> Tuple[str, bytes]:
        with self.lock:
            summary = self._require_book_summary(slug)
            if not summary.open_enabled:
                raise ValueError("Generate Voices before sharing this book.")
            archive = build_readalong_package(summary.book_root)
            return f"{summary.slug}.readalong.zip", archive

    def import_book_package(self, upload: UploadedReadAlongPackage) -> Dict[str, Any]:
        with self.lock:
            if not upload.content:
                raise ValueError("ReadAlong package upload is required.")
            target_root = import_readalong_package(
                self.library_root,
                upload.content,
                requested_slug=upload.slug,
            )
            summary = summarize_book(target_root)
            return {
                "ok": True,
                "book": summary.to_payload(),
                "active_book": self._active_book_payload(),
                "library": self.library_payload(),
            }

    def process_book(self) -> Dict[str, Any]:
        with self.lock:
            result = self._require_controller().process_read_along_book()
            return {"ok": True, **result}

    def build_units(self, chapter: str) -> Dict[str, Any]:
        with self.lock:
            units = self._require_controller().build_read_along_units(chapter)
            return {"ok": True, "chapter": chapter, "units": units, "unit_count": len(units)}

    def save_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.session is not None:
                raise ValueError("End the active read-along session before changing settings.")
            controller = self._require_controller()
            controller.save_read_along_settings(settings)
            return {"ok": True, "settings": controller.read_along_settings()}

    def narrator_profile_payload(self) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            return {"ok": True, **controller.read_along_narrator_profile_payload()}

    def save_narrator_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.session is not None:
                raise ValueError("End the active read-along session before editing narrator.")
            controller = self._require_controller()
            controller.save_read_along_narrator_profile(payload)
            return {"ok": True, **controller.read_along_narrator_profile_payload()}

    def audiobook_narrator_profile_payload(self) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            return {"ok": True, **controller.audiobook_narrator_profile_payload()}

    def save_audiobook_narrator_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            controller = self._require_controller()
            controller.save_audiobook_narrator_profile(payload)
            return {"ok": True, **controller.audiobook_narrator_profile_payload()}

    def save_reading_position(self, slug: str, chapter: str, unit_id: int) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            _validate_reading_position(summary.book_root, chapter, unit_id)
            _update_book_last_read(summary.book_root, chapter=chapter, unit_id=int(unit_id))
            last_read = {"chapter": chapter, "unit_id": int(unit_id)}
            return {
                "ok": True,
                "slug": summary.slug,
                "last_read": last_read,
                "last_read_display": _last_read_label(last_read),
                "library": self.library_payload(),
            }

    def audiobook_payload(self, slug: str) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(slug)
            controller = self._make_controller(summary.book_root)
            paths = controller.paths
            manifest = _read_json_if_exists(paths.audiobook_manifest)
            manifest_chapters = manifest.get("chapters", {}) if isinstance(manifest.get("chapters"), dict) else {}
            chapters = []
            for row in controller.chapter_rows():
                entry = dict(manifest_chapters.get(row.chapter, {}))
                audio_path = paths.audiobook_chapter_audio(row.chapter)
                timeline_path = paths.audiobook_chapter_timeline(row.chapter)
                audio_ready = audio_path.exists()
                chapters.append(
                    {
                        "chapter": row.chapter,
                        "index": row.index,
                        "title": row.title,
                        "audio_ready": audio_ready,
                        "timeline_ready": timeline_path.exists(),
                        "audio_url": f"/api/audiobook/audio/{row.chapter}.wav?slug={summary.slug}" if audio_ready else "",
                        "duration_seconds": float(entry.get("duration_seconds") or 0.0),
                        "generated_at": entry.get("generated_at"),
                        "window_count": int(entry.get("window_count") or 0),
                        "unit_count": int(entry.get("unit_count") or 0),
                    }
                )
            job = self._running_job_for_slug(summary.slug)
            return {
                "ok": True,
                "book": summary.to_payload(active=self.controller is not None and _same_path(self.controller.book_root, summary.book_root)),
                "chapters": chapters,
                "settings": controller.audiobook_settings(),
                "position": _read_json_if_exists(paths.audiobook_position),
                "job": job.to_payload() if job is not None and job.action_key == "audiobook" else None,
            }

    def generate_audiobook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(str(payload.get("slug", "")))
            if not summary.open_enabled:
                raise ValueError("Generate Voices before generating audiobook audio.")
            existing = self._running_job_for_slug(summary.slug)
            if existing is not None:
                return {
                    "ok": True,
                    "book": summary.to_payload(),
                    "job": existing.to_payload(),
                    "library": self.library_payload(),
                }
            controller = self._make_controller(summary.book_root)
            requested = [str(chapter).strip() for chapter in payload.get("chapters", []) if str(chapter).strip()]
            if not requested:
                requested = [row.chapter for row in controller.chapter_rows()]
            valid = {row.chapter for row in controller.chapter_rows()}
            chapters = [chapter for chapter in requested if chapter in valid]
            if not chapters:
                raise ValueError("Select at least one valid chapter.")
            job = LibraryJob(
                job_id=uuid.uuid4().hex,
                slug=summary.slug,
                action_key="audiobook",
                completed=0,
                total=len(chapters),
                chapters=chapters,
                settings=dict(payload.get("settings", {})),
                force=bool(payload.get("force")),
            )
            self.jobs[job.job_id] = job
            self.jobs_by_slug[summary.slug] = job.job_id
            worker = threading.Thread(target=self._run_audiobook_job, args=(job.job_id,), daemon=True)
            worker.start()
            return {
                "ok": True,
                "book": summary.to_payload(),
                "job": job.to_payload(),
                "library": self.library_payload(),
            }

    def _run_audiobook_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            summary = self._require_book_summary(job.slug)
            controller = self._make_controller(summary.book_root)
            job.status = "running"

        def progress_callback(event: Dict[str, Any]) -> None:
            with self.lock:
                current = self.jobs[job_id]
                current.total = int(event.get("total") or current.total or 0)
                current.current_chapter = str(event.get("chapter") or current.current_chapter)
                current.current_item = f"{int(event.get('window_count') or 0)} windows"
                status = str(event.get("status") or "")
                if status == "started":
                    current.completed = max(0, int(event.get("completed") or 0))
                if status == "completed":
                    current.completed = max(current.completed, int(event.get("completed") or current.completed))

        try:
            result = controller.generate_audiobook_chapters(
                job.chapters,
                job.settings,
                force=job.force,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                job.status = "failed"
                job.failed_chapter = job.current_chapter
                job.error = str(exc)
                job.finished_at = time.time()
                summary = self._require_book_summary(job.slug)
                self.log_event(
                    "library_audiobook_job_error",
                    {
                        "job_id": job_id,
                        "slug": job.slug,
                        "current_chapter": job.current_chapter,
                    },
                    exc=exc,
                    book_root=summary.book_root,
                )
            return

        with self.lock:
            job = self.jobs[job_id]
            job.status = "completed"
            job.completed = int(result.get("chapters", job.completed))
            job.total = int(result.get("chapters", job.total))
            job.current_chapter = ""
            job.current_item = ""
            job.finished_at = time.time()

    def audiobook_audio(self, slug: str, chapter: str) -> bytes:
        with self.lock:
            summary = self._require_book_summary(slug)
            paths = BookPaths(summary.book_root)
            audio_root = (paths.root / "audiobook").resolve()
            audio_path = paths.audiobook_chapter_audio(chapter).resolve()
            if audio_root not in audio_path.parents:
                raise FileNotFoundError("audiobook audio path is outside the audiobook directory")
            return audio_path.read_bytes()

    def save_audiobook_position(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            summary = self._require_book_summary(str(payload.get("slug", "")))
            chapter = str(payload.get("chapter", "")).strip()
            if not chapter:
                raise ValueError("chapter is required")
            paths = BookPaths(summary.book_root)
            position = {
                "chapter": chapter,
                "position_seconds": max(0.0, float(payload.get("position_seconds") or 0.0)),
                "updated_at": time.time(),
            }
            write_json_atomic(paths.audiobook_position, position)
            return {"ok": True, "position": position}

    def start_session(
        self,
        chapter: str,
        start_unit_id: int,
        settings: Dict[str, Any],
        reuse_active_tts: bool = False,
    ) -> Dict[str, Any]:
        self._set_session_start_progress("validating_chapter", "Validating chapter and read-along units.")
        try:
            if not chapter.strip():
                raise ValueError("chapter is required")
            with self.lock:
                tts_adapter = None
                if reuse_active_tts and self.session is not None:
                    tts_adapter = self.session.tts_adapter
                    self.session.end(close_adapter=False)
                    self.session = None
                    self.session_chapter = ""
                else:
                    self.end_session()
                controller = self._require_controller()
                if not controller.paths.annotation(chapter).exists() and not controller.paths.read_along_units(chapter).exists():
                    raise ValueError("Process Book before starting read-along for this chapter.")
                self._set_session_start_progress("saving_settings", "Saving read-along session settings.")
                controller.save_read_along_settings(settings)
                resolved_settings = controller.read_along_settings()
                units = controller.read_along_units(chapter)
                if not units:
                    raise ValueError("Process Book before starting read-along for this chapter.")
                log_runtime_step(
                    "web_readalong_session_start",
                    chapter=chapter,
                    start_unit_id=start_unit_id,
                    reuse_active_tts=reuse_active_tts and tts_adapter is not None,
                    playback_speed=resolved_settings.get("playback_speed"),
                    start_buffer_seconds=resolved_settings.get("target_buffer_seconds"),
                    target_buffer_seconds=resolved_settings.get("target_buffer_seconds"),
                    generation_mode=resolved_settings.get("generation_mode"),
                )
                if tts_adapter is None:
                    self._set_session_start_progress("loading_tts_model", "Loading read-along TTS model.")
                else:
                    self._set_session_start_progress("reusing_tts_model", "Reusing warm read-along TTS model.")
                self.session = controller.create_read_along_session(
                    chapter,
                    units,
                    resolved_settings,
                    store_audio_files=False,
                    progress_callback=lambda event: self._set_session_start_progress(
                        str(event.get("stage") or "starting_session"),
                        str(event.get("message") or "Starting read-along session."),
                    ),
                    tts_adapter=tts_adapter,
                )
                self.session_chapter = chapter
                self._set_session_start_progress(
                    "loading_tts_stack",
                    (
                        "Loading TTS stack and generating the first audio buffer: "
                        f"0.0 / {self.session.start_buffer_seconds:.1f}s."
                    ),
                )
                with self.session_fill_lock:
                    self.session.fill_buffer(
                        start_unit_id=start_unit_id,
                        min_buffer_seconds=self.session.start_buffer_seconds,
                        progress_callback=lambda event: self._set_session_start_progress(
                            str(event.get("stage") or "building_initial_buffer"),
                            (
                                "Building initial audio buffer: "
                                f"{float(event.get('ready_playback_seconds') or 0.0):.1f} / "
                                f"{float(event.get('target_buffer_seconds') or self.session.start_buffer_seconds):.1f}s."
                            ),
                        ),
                        progress_stage="building_initial_buffer",
                    )
                self._record_last_read(chapter, start_unit_id)
                ready_seconds = self.session.ready_playback_seconds
                self._set_session_start_progress(
                    "buffer_ready",
                    f"Buffer ready: {ready_seconds:.1f}s of generated audio.",
                    status="completed",
                )
                log_runtime_step(
                    "web_readalong_buffer_ready",
                    session_id=self.session.session_id,
                    ready_units=len(self.session.ready_items),
                    ready_playback_seconds=f"{ready_seconds:.2f}",
                )
                return {
                    "ok": True,
                    "chapter": chapter,
                    "settings": resolved_settings,
                    "session_id": self.session.session_id,
                    "units": [unit.to_dict() for unit in self.session.units],
                    "ready": self._ready_payload(),
                    "ready_playback_seconds": ready_seconds,
                    "target_buffer_seconds": self.session.target_buffer_seconds,
                    "max_buffer_seconds": self.session.max_buffer_seconds,
                    "has_more_units": self.session.has_more_units,
                }
        except Exception as exc:
            self._set_session_start_progress("failed", str(exc), status="failed")
            self.log_event(
                "readalong_session_start_error",
                {
                    "chapter": chapter,
                    "start_unit_id": start_unit_id,
                    "reuse_active_tts": reuse_active_tts,
                },
                exc=exc,
            )
            raise

    def advance_session(self) -> Dict[str, Any]:
        with self.lock:
            session = self._require_session()
            consumed = session.consume_ready()
            if session.ready_items:
                self._record_last_read(self.session_chapter, session.ready_items[0].unit_id)
            elif consumed is not None:
                self._record_last_read(self.session_chapter, consumed.unit_id)
            return {
                "ok": True,
                "chapter": self.session_chapter,
                "session_id": session.session_id,
                "ready": self._ready_payload(),
                "ready_playback_seconds": session.ready_playback_seconds,
                "target_buffer_seconds": session.target_buffer_seconds,
                "max_buffer_seconds": session.max_buffer_seconds,
                "has_more_units": session.has_more_units,
                "ended": not session.ready_items and not session.has_more_units,
            }

    def top_up_session(self, exclude_unit_id: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            session = self._require_session()
            session_id = session.session_id
            start_details = {
                "chapter": self.session_chapter,
                "session_id": session_id,
                "exclude_unit_id": exclude_unit_id,
                "ready_unit_ids": session.ready_unit_ids,
                "ready_playback_seconds": session.ready_playback_seconds,
                "has_more_units": session.has_more_units,
                "next_unit_id": getattr(session, "_next_unit_id", None),
            }
        self.log_event("readalong_top_up_start", start_details)
        if not self.session_fill_lock.acquire(blocking=False):
            with self.lock:
                current = self._require_session()
                if current.session_id != session_id:
                    raise ValueError("Read-along session changed while topping up buffer.")
                self.log_event(
                    "readalong_top_up_already_running",
                    {
                        "chapter": self.session_chapter,
                        "session_id": current.session_id,
                        "ready_unit_ids": current.ready_unit_ids,
                        "ready_playback_seconds": current.ready_playback_seconds,
                    },
                )
                return {
                    "ok": True,
                    "chapter": self.session_chapter,
                    "session_id": current.session_id,
                    "ready": self._ready_payload(),
                    "ready_playback_seconds": current.ready_playback_seconds,
                    "target_buffer_seconds": current.target_buffer_seconds,
                    "max_buffer_seconds": current.max_buffer_seconds,
                    "has_more_units": current.has_more_units,
                    "generated_count": 0,
                    "top_up_running": True,
                }
        try:
            generated = session.fill_buffer(exclude_unit_id=exclude_unit_id)
        except Exception as exc:
            self.log_event("readalong_top_up_error", start_details, exc=exc)
            raise
        finally:
            self.session_fill_lock.release()
        with self.lock:
            current = self._require_session()
            if current.session_id != session_id:
                raise ValueError("Read-along session changed while topping up buffer.")
            self.log_event(
                "readalong_top_up_done",
                {
                    "chapter": self.session_chapter,
                    "session_id": current.session_id,
                    "generated_count": len(generated),
                    "generated_unit_ids": [item.unit_id for item in generated],
                    "ready_unit_ids": current.ready_unit_ids,
                    "ready_playback_seconds": current.ready_playback_seconds,
                    "has_more_units": current.has_more_units,
                    "next_unit_id": getattr(current, "_next_unit_id", None),
                },
            )
            return {
                "ok": True,
                "chapter": self.session_chapter,
                "session_id": current.session_id,
                "ready": self._ready_payload(),
                "ready_playback_seconds": current.ready_playback_seconds,
                "target_buffer_seconds": current.target_buffer_seconds,
                "max_buffer_seconds": current.max_buffer_seconds,
                "has_more_units": current.has_more_units,
                "generated_count": len(generated),
                "top_up_running": False,
            }

    def end_session(self) -> Dict[str, Any]:
        with self.lock:
            if self.session is not None:
                self.session.end()
            self.session = None
            self.session_chapter = ""
            return {"ok": True}

    def audio_for_unit(self, session_id: str, unit_id: int) -> Tuple[bytes, float]:
        with self.lock:
            session = self._require_session()
            if session.session_id != session_id:
                raise FileNotFoundError("session not found")
            for item in session.ready_items:
                if item.unit_id == unit_id:
                    if item.audio_bytes:
                        return item.audio_bytes, item.playback_seconds
                    if item.audio_path is None:
                        raise FileNotFoundError("audio bytes are no longer ready")
                    path = item.audio_path.resolve()
                    root = session.session_dir.resolve()
                    if root not in path.parents and path != root:
                        raise FileNotFoundError("audio path is outside the session directory")
                    if not path.exists():
                        raise FileNotFoundError("audio file is no longer ready")
                    return path.read_bytes(), item.playback_seconds
            raise FileNotFoundError("audio unit is not ready")

    def _ready_payload(self) -> List[Dict[str, Any]]:
        session = self._require_session()
        return [
            {
                "ready_index": index,
                "unit_id": item.unit_id,
                "audio_url": f"/api/session/{session.session_id}/audio/{item.unit_id}.wav",
                "playback_seconds": item.playback_seconds,
            }
            for index, item in enumerate(session.ready_items)
        ]

    def _require_session(self) -> ReadAlongSession:
        if self.session is None:
            raise ValueError("No active read-along session.")
        return self.session

    def _require_controller(self) -> PrototypeUiController:
        if self.controller is None:
            raise ValueError("Select or add a book before using the reader.")
        return self.controller

    def _make_controller(self, book_root: Path) -> PrototypeUiController:
        resolved_book_root = Path(book_root).resolve()
        return PrototypeUiController(
            book_root=resolved_book_root,
            pipeline_factory=self.pipeline_factory,
            extractor=self.extractor,
            fake_tts=self.fake_tts,
            library_path=self.library_root / "library.json",
        )

    def _active_book_payload(self) -> Optional[Dict[str, Any]]:
        if self.controller is None:
            return None
        return summarize_book(self.controller.book_root).to_payload(active=True)

    def _book_root_for_slug(self, slug: str) -> Path:
        folder = _safe_slug(slug)
        library_root = self.library_root.resolve()
        target = (library_root / folder).resolve()
        try:
            relative = target.relative_to(library_root)
        except ValueError as exc:
            raise ValueError("Folder name must stay inside the library root.") from exc
        if len(relative.parts) != 1:
            raise ValueError("Folder name must be a single folder name.")
        return target

    def _require_book_summary(self, slug: str) -> LibraryBookSummary:
        normalized = str(slug).strip()
        if not normalized:
            raise ValueError("slug is required")
        for summary in discover_books(self.library_root):
            if summary.slug == normalized:
                return summary
        raise ValueError(f"Book not found: {slug}")

    def _record_last_read(self, chapter: str, unit_id: int) -> None:
        controller = self._require_controller()
        _update_book_last_read(controller.book_root, chapter=chapter, unit_id=int(unit_id))

    def _set_session_start_progress(self, stage: str, message: str, status: str = "running") -> None:
        with self.progress_lock:
            self.session_start_progress = {
                "status": status,
                "stage": str(stage),
                "message": str(message),
                "updated_at": time.time(),
            }


class ReadAlongHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, app_state: ReadAlongWebState):
        super().__init__(server_address, request_handler_class)
        self.app_state = app_state


def create_server(
    book_root: Optional[str | Path] = None,
    launch_root: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    fake_tts: bool = False,
    extractor: Optional[ChapterExtractor] = None,
    pipeline_factory: Optional[PipelineFactory] = None,
) -> ReadAlongHttpServer:
    if book_root is not None:
        root = Path(book_root).resolve()
        state = ReadAlongWebState(
            library_root=root.parent,
            fake_tts=fake_tts,
            extractor=extractor,
            pipeline_factory=pipeline_factory,
        )
        state.controller = state._make_controller(root)
    else:
        selection = resolve_launch_root(Path.cwd() if launch_root is None else launch_root)
        state = ReadAlongWebState(
            library_root=selection.library_root,
            fake_tts=fake_tts,
            extractor=extractor,
            pipeline_factory=pipeline_factory,
        )
        if selection.active_book_root is not None:
            state.controller = state._make_controller(selection.active_book_root)
    return ReadAlongHttpServer((host, int(port)), build_handler(state), state)


def build_handler(app_state: ReadAlongWebState):
    class ReadAlongRequestHandler(BaseHTTPRequestHandler):
        server_version = "EbookReadAlong/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/":
                    self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
                    return
                if path == "/api/library":
                    self._send_json(app_state.library_payload())
                    return
                if path == "/api/library/job-status":
                    job_id = parse_qs(parsed.query).get("job_id", [""])[0]
                    self._send_json(app_state.job_status(job_id))
                    return
                if path == "/api/library/export":
                    slug = parse_qs(parsed.query).get("slug", [""])[0]
                    filename, archive = app_state.export_book_package(slug)
                    self._send_bytes(
                        archive,
                        content_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                    )
                    return
                if path == "/api/registry":
                    slug = parse_qs(parsed.query).get("slug", [""])[0]
                    self._send_json(app_state.registry_review(slug))
                    return
                if path.startswith("/api/registry/sample/"):
                    role_file = path.rsplit("/", 1)[-1]
                    role_id = role_file[:-4] if role_file.endswith(".wav") else role_file
                    slug = parse_qs(parsed.query).get("slug", [""])[0]
                    audio_bytes = app_state.registry_sample_audio(role_id, slug=slug)
                    self._send_bytes(audio_bytes, content_type="audio/wav")
                    return
                if path == "/api/audiobook":
                    slug = parse_qs(parsed.query).get("slug", [""])[0]
                    self._send_json(app_state.audiobook_payload(slug))
                    return
                if path == "/api/audiobook/narrator-profile":
                    self._send_json(app_state.audiobook_narrator_profile_payload())
                    return
                audiobook_audio_match = _audiobook_audio_path_parts(path)
                if audiobook_audio_match is not None:
                    slug = parse_qs(parsed.query).get("slug", [""])[0]
                    self._send_bytes(app_state.audiobook_audio(slug, audiobook_audio_match), content_type="audio/wav")
                    return
                if path == "/api/state":
                    self._send_json(app_state.state_payload())
                    return
                if path == "/api/session/start-progress":
                    self._send_json(app_state.session_start_progress_payload())
                    return
                if path == "/api/narrator-profile":
                    self._send_json(app_state.narrator_profile_payload())
                    return
                if path.startswith("/api/chapter/"):
                    chapter = path.rsplit("/", 1)[-1]
                    self._send_json(app_state.chapter_payload(chapter))
                    return
                audio_match = _audio_path_parts(path)
                if audio_match is not None:
                    session_id, unit_id = audio_match
                    audio_bytes, _ = app_state.audio_for_unit(session_id, unit_id)
                    self._send_bytes(audio_bytes, content_type="audio/wav")
                    return
                self._send_json_error(HTTPStatus.NOT_FOUND, "Not found.")
            except Exception as exc:
                app_state.log_event(
                    "http_get_error",
                    {"path": path, "query": parsed.query},
                    exc=exc,
                )
                self._send_json_error(_status_for_exception(exc), str(exc))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/api/library/add-book" and self._is_multipart_request():
                    self._send_json(app_state.add_uploaded_book(self._read_upload_body()))
                    return
                if path == "/api/library/import-package" and self._is_multipart_request():
                    self._send_json(app_state.import_book_package(self._read_package_upload_body()))
                    return
                payload = self._read_json_body()
                if path == "/api/library/select":
                    self._send_json(app_state.select_book(str(payload.get("slug", ""))))
                    return
                if path == "/api/library/add-book":
                    self._send_json(app_state.add_book(payload))
                    return
                if path == "/api/library/initialize":
                    self._send_json(app_state.initialize_book(str(payload.get("slug", ""))))
                    return
                if path == "/api/library/build-registry":
                    self._send_json(app_state.build_book_registry(str(payload.get("slug", ""))))
                    return
                if path == "/api/library/annotate":
                    self._send_json(app_state.annotate_book(str(payload.get("slug", ""))))
                    return
                if path == "/api/library/prepare-voices":
                    self._send_json(app_state.prepare_book_voices(str(payload.get("slug", ""))))
                    return
                if path == "/api/library/delete":
                    self._send_json(app_state.delete_book(str(payload.get("slug", ""))))
                    return
                if path == "/api/registry/save-character":
                    self._send_json(app_state.registry_save_character(payload))
                    return
                if path == "/api/registry/save-characters":
                    self._send_json(app_state.registry_save_characters(payload))
                    return
                if path == "/api/registry/confirm":
                    self._send_json(app_state.registry_confirm(str(payload.get("slug", ""))))
                    return
                if path == "/api/registry/generate-sample":
                    self._send_json(app_state.registry_generate_sample(payload))
                    return
                if path == "/api/audiobook/generate":
                    self._send_json(app_state.generate_audiobook(payload))
                    return
                if path == "/api/audiobook/narrator-profile":
                    self._send_json(app_state.save_audiobook_narrator_profile(payload))
                    return
                if path == "/api/audiobook/position":
                    self._send_json(app_state.save_audiobook_position(payload))
                    return
                if path == "/api/process-book":
                    self._send_json(app_state.process_book())
                    return
                if path == "/api/build-units":
                    self._send_json(app_state.build_units(str(payload.get("chapter", ""))))
                    return
                if path == "/api/settings":
                    self._send_json(app_state.save_settings(payload))
                    return
                if path == "/api/narrator-profile":
                    self._send_json(app_state.save_narrator_profile(payload))
                    return
                if path == "/api/reading-position":
                    self._send_json(
                        app_state.save_reading_position(
                            str(payload.get("slug", "")),
                            str(payload.get("chapter", "")),
                            int(payload.get("unit_id", 0) or 0),
                        )
                    )
                    return
                if path == "/api/session/start":
                    self._send_json(
                        app_state.start_session(
                            chapter=str(payload.get("chapter", "")),
                            start_unit_id=int(payload.get("start_unit_id", 0) or 0),
                            settings=dict(payload.get("settings", {})),
                            reuse_active_tts=bool(payload.get("reuse_active_tts")),
                        )
                    )
                    return
                if path == "/api/session/advance":
                    self._send_json(app_state.advance_session())
                    return
                if path == "/api/session/top-up":
                    exclude_unit_id = payload.get("exclude_unit_id")
                    self._send_json(
                        app_state.top_up_session(
                            None if exclude_unit_id is None else int(exclude_unit_id)
                        )
                    )
                    return
                if path == "/api/session/end":
                    self._send_json(app_state.end_session())
                    return
                self._send_json_error(HTTPStatus.NOT_FOUND, "Not found.")
            except Exception as exc:
                app_state.log_event(
                    "http_post_error",
                    {
                        "path": path,
                        "payload_keys": sorted(str(key) for key in payload.keys()) if "payload" in locals() else [],
                    },
                    exc=exc,
                )
                self._send_json_error(_status_for_exception(exc), str(exc))

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return dict(json.loads(raw.decode("utf-8")))

        def _is_multipart_request(self) -> bool:
            content_type = self.headers.get("Content-Type", "")
            return content_type.lower().startswith("multipart/form-data")

        def _read_upload_body(self) -> UploadedBook:
            form = self._read_multipart_form("epub upload is required")
            epub_field = form["epub"] if "epub" in form else None
            if epub_field is None or not getattr(epub_field, "filename", ""):
                raise ValueError("epub upload is required")
            content = epub_field.file.read()
            return UploadedBook(
                filename=str(epub_field.filename),
                content=content,
                title=_multipart_value(form, "title"),
                author=_multipart_value(form, "author"),
                slug=_multipart_value(form, "slug"),
            )

        def _read_package_upload_body(self) -> UploadedReadAlongPackage:
            form = self._read_multipart_form("ReadAlong package upload is required")
            package_field = form["package"] if "package" in form else None
            if package_field is None or not getattr(package_field, "filename", ""):
                raise ValueError("ReadAlong package upload is required")
            return UploadedReadAlongPackage(
                filename=str(package_field.filename),
                content=package_field.file.read(),
                slug=_multipart_value(form, "slug"),
            )

        def _read_multipart_form(self, empty_message: str) -> cgi.FieldStorage:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                raise ValueError(empty_message)
            body = self.rfile.read(length)
            return cgi.FieldStorage(
                fp=io.BytesIO(body),
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": str(length),
                },
                keep_blank_values=True,
            )

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status=status)

        def _send_text(self, text: str, content_type: str) -> None:
            self._send_bytes(text.encode("utf-8"), content_type=content_type)

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            headers: Optional[Dict[str, str]] = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ReadAlongRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readalongweb")
    parser.add_argument("launch_root", nargs="?", default=".")
    parser.add_argument("--root", dest="root", default=None)
    parser.add_argument("--book-root", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--fake-tts", action="store_true")
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=True)
    return parser


def run_server(
    book_root: Optional[str | Path] = None,
    launch_root: Optional[str | Path] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    fake_tts: bool = False,
    open_browser: bool = True,
) -> int:
    try:
        server = create_server(
            book_root=book_root,
            launch_root=Path.cwd() if launch_root is None else launch_root,
            host=host,
            port=port,
            fake_tts=fake_tts,
        )
    except OSError:
        if int(port) == 0:
            raise
        server = create_server(
            book_root=book_root,
            launch_root=Path.cwd() if launch_root is None else launch_root,
            host=host,
            port=0,
            fake_tts=fake_tts,
        )
    resolved_host, resolved_port = server.server_address
    url = f"http://{resolved_host}:{resolved_port}/"
    print(f"Read-along web UI: {url}", flush=True)
    _install_process_crash_logging(server.app_state)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        server.app_state.log_event("readalong_web_server_crash", {"url": url}, exc=exc)
        raise
    finally:
        server.app_state.end_session()
        server.server_close()
    return 0


def _install_process_crash_logging(app_state: ReadAlongWebState) -> None:
    previous_excepthook = sys.excepthook
    previous_threading_excepthook = getattr(threading, "excepthook", None)

    def excepthook(exc_type, exc, tb):
        app_state.log_event("process_uncaught_exception", {"exception_type": getattr(exc_type, "__name__", str(exc_type))}, exc=exc)
        previous_excepthook(exc_type, exc, tb)

    def threading_excepthook(args):
        app_state.log_event(
            "thread_uncaught_exception",
            {
                "thread_name": getattr(args.thread, "name", ""),
                "exception_type": getattr(args.exc_type, "__name__", str(args.exc_type)),
            },
            exc=args.exc_value,
        )
        if previous_threading_excepthook is not None:
            previous_threading_excepthook(args)

    sys.excepthook = excepthook
    if previous_threading_excepthook is not None:
        threading.excepthook = threading_excepthook


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    launch_root = args.root if args.root is not None else args.launch_root
    return run_server(
        book_root=args.book_root,
        launch_root=launch_root,
        host=args.host,
        port=args.port,
        fake_tts=args.fake_tts,
        open_browser=args.open_browser,
    )


def resolve_launch_root(launch_root: str | Path) -> LaunchSelection:
    root = Path(launch_root).resolve()
    if _is_book_root(root):
        return LaunchSelection(library_root=root.parent, active_book_root=root)
    books_dir = root / "books"
    if books_dir.is_dir():
        return LaunchSelection(library_root=books_dir, active_book_root=None)
    return LaunchSelection(library_root=root, active_book_root=None)


def discover_books(library_root: str | Path) -> List[LibraryBookSummary]:
    root = Path(library_root).resolve()
    if not root.exists():
        return []
    books = [
        summarize_book(path)
        for path in root.iterdir()
        if path.is_dir() and (_is_book_root(path) or _is_manifest_book_root(path))
    ]
    return sorted(books, key=lambda book: (book.title.lower(), book.slug.lower()))


def summarize_book(book_root: str | Path) -> LibraryBookSummary:
    root = Path(book_root).resolve()
    manifest = _read_book_manifest(root)
    registry = _read_json_if_exists(root / "registry.json")
    title = ""
    author = ""
    if isinstance(registry.get("book"), dict):
        title = str(registry["book"].get("title", "")).strip()
        author = str(registry["book"].get("author", "")).strip()
    title = title or str(manifest.get("title", "")).strip()
    author = author or str(manifest.get("author", "")).strip()
    title = title or root.name.replace("_", " ").replace("-", " ").title() or "Untitled Book"
    chapter_count = _glob_count(root / "chapters", "*.txt")
    annotation_count, unit_count = _read_along_readiness_counts(root)
    voice_count, voice_total = _registry_voice_readiness(root, registry)
    audio_count = _chapter_audio_count(root)
    status_key, status_label = _book_status(
        chapter_count=chapter_count,
        annotation_count=annotation_count,
        read_along_unit_count=unit_count,
        voice_count=voice_count,
        voice_total=voice_total,
        audio_count=audio_count,
        has_registry=(root / "registry.json").exists(),
        manifest=manifest,
    )
    action_key, action_label, open_enabled = _book_action(status_key)
    resume_annotation_enabled, resume_annotation_label = _resume_annotation_action(
        book_root=root,
        status_key=status_key,
        has_registry=(root / "registry.json").exists(),
    )
    last_read = _manifest_last_read(manifest)
    return LibraryBookSummary(
        title=title,
        author=author,
        slug=root.name,
        book_root=root,
        chapter_count=chapter_count,
        annotation_count=annotation_count,
        read_along_unit_count=unit_count,
        voice_count=voice_count,
        voice_total=voice_total,
        audio_count=audio_count,
        has_registry=(root / "registry.json").exists(),
        status_key=status_key,
        status_label=status_label,
        status_detail=(
            f"{chapter_count} chapters, {annotation_count}/{chapter_count} annotated, "
            f"{unit_count}/{chapter_count} read-along units, {voice_count}/{voice_total} voices, "
            f"{audio_count} audio files"
        ),
        action_key=action_key,
        action_label=action_label,
        open_enabled=open_enabled,
        resume_annotation_enabled=resume_annotation_enabled,
        resume_annotation_label=resume_annotation_label,
        last_read=last_read,
        last_read_label=_last_read_label(last_read),
    )


def _audio_path_parts(path: str) -> Optional[Tuple[str, int]]:
    prefix = "/api/session/"
    suffix = ".wav"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    parts = path[len(prefix):].split("/")
    if len(parts) != 3 or parts[1] != "audio":
        return None
    return parts[0], int(parts[2][:-len(suffix)])


def _audiobook_audio_path_parts(path: str) -> Optional[str]:
    prefix = "/api/audiobook/audio/"
    suffix = ".wav"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    chapter = path[len(prefix):-len(suffix)]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chapter):
        return None
    return chapter


def _chapter_audio_count(book_root: Path) -> int:
    chapters = set()
    for folder in ("audio", "audiobook"):
        audio_dir = book_root / folder
        if not audio_dir.exists():
            continue
        chapters.update(path.stem for path in audio_dir.glob("*.wav"))
    return len(chapters)


def _status_for_exception(exc: Exception) -> HTTPStatus:
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return HTTPStatus.BAD_REQUEST
    if isinstance(exc, (FileNotFoundError, OSError)):
        return HTTPStatus.NOT_FOUND
    return HTTPStatus.INTERNAL_SERVER_ERROR


def _is_book_root(path: Path) -> bool:
    chapters = path / "chapters"
    return chapters.is_dir() and any(chapters.glob("*.txt"))


def _is_manifest_book_root(path: Path) -> bool:
    return (path / BOOK_MANIFEST).exists()


def _book_status(
    chapter_count: int,
    annotation_count: int,
    read_along_unit_count: int,
    voice_count: int,
    voice_total: int,
    audio_count: int,
    has_registry: bool,
    manifest: Dict[str, Any],
) -> Tuple[str, str]:
    stages = dict(manifest.get("stages", {})) if isinstance(manifest.get("stages"), dict) else {}
    annotations_ready = chapter_count > 0 and annotation_count >= chapter_count and read_along_unit_count >= chapter_count
    if manifest:
        if not stages.get("initialized"):
            return "fresh_added", "Freshly added"
        if not stages.get("global_registry"):
            return "initialized", "Initialized"
        if stages.get("annotating"):
            return "annotating", "Annotating"
        if not stages.get("annotated"):
            if annotation_count > 0 or read_along_unit_count > 0:
                return "partially_annotated", "Partially annotated"
            return "registry_ready", "Registry ready"
        if not annotations_ready:
            if annotation_count > 0 or read_along_unit_count > 0:
                return "partially_annotated", "Partially annotated"
            return "registry_ready", "Registry ready"
        voices_current = voice_total > 0 and voice_count >= voice_total
        if not stages.get("voices_ready") or not voices_current:
            return "annotated", "Annotated"
        return "voices_ready", "Voices ready"
    if chapter_count <= 0:
        return "not_initialized", "Not initialized"
    if audio_count >= chapter_count:
        return "audio_ready", "Audio ready"
    if read_along_unit_count >= chapter_count and voice_count >= voice_total:
        return "voices_ready", "Voices ready"
    if read_along_unit_count >= chapter_count:
        return "annotated", "Annotated"
    if annotation_count >= chapter_count:
        return "annotated", "Annotated"
    if annotation_count > 0:
        return "partially_annotated", "Partially annotated"
    if has_registry:
        return "registry_ready", "Registry ready"
    return "initialized", "Initialized"


def _book_action(status_key: str) -> Tuple[str, str, bool]:
    if status_key == "fresh_added":
        return "initialize", "Initialize Book", False
    if status_key == "initialized":
        return "build_registry", "Build Registry", False
    if status_key == "annotating":
        return "annotate", "Annotating", False
    if status_key in {"registry_ready", "partially_annotated"}:
        return "annotate", "Annotate Book", False
    if status_key in {"registry_review", "registry_reviewed", "annotated"}:
        return "prepare_voices", "Generate Voices", False
    if status_key in {"voices_ready", "audio_ready"}:
        return "review_registry", "Review Voices", True
    return "initialize", "Initialize Book", False


def _resume_annotation_action(book_root: Path, status_key: str, has_registry: bool) -> Tuple[bool, str]:
    if not has_registry or status_key in {"fresh_added", "initialized", "annotating"}:
        return False, ""
    progress = _read_json_if_exists(book_root / "read_along" / "annotation_progress.json")
    if progress.get("status") == "failed":
        return True, "Retry Annotation"
    if progress.get("status") == "running":
        return True, "Resume Annotation"
    return False, ""


def _read_book_manifest(book_root: Path) -> Dict[str, Any]:
    payload = _read_json_if_exists(book_root / BOOK_MANIFEST)
    return payload if isinstance(payload, dict) else {}


def _write_book_manifest(book_root: Path, title: str, author: str, slug: str, original_filename: str) -> None:
    source_path = SOURCE_EPUB_PATH.as_posix()
    write_json_atomic(
        book_root / BOOK_MANIFEST,
        {
            "schema": "readalong_book_v1",
            "title": title,
            "author": author,
            "slug": slug,
            "source_epub_path": source_path,
            "original_filename": original_filename,
            "stages": {
                "source_added": True,
                "initialized": False,
                "global_registry": False,
                "annotating": False,
                "annotated": False,
                "registry_reviewed": False,
                "voices_ready": False,
            },
            "last_read": {},
        },
    )


def _update_book_stage(
    book_root: Path,
    initialized: Optional[bool] = None,
    global_registry: Optional[bool] = None,
    annotating: Optional[bool] = None,
    annotated: Optional[bool] = None,
    registry_reviewed: Optional[bool] = None,
    voices_ready: Optional[bool] = None,
) -> None:
    manifest = _read_book_manifest(book_root)
    if not manifest:
        return
    stages = dict(manifest.get("stages", {}))
    if initialized is not None:
        stages["initialized"] = bool(initialized)
        if not initialized:
            stages["global_registry"] = False
            stages["annotating"] = False
            stages["annotated"] = False
            stages["registry_reviewed"] = False
            stages["voices_ready"] = False
    if global_registry is not None:
        stages["global_registry"] = bool(global_registry)
        if not global_registry:
            stages["annotating"] = False
            stages["annotated"] = False
            stages["registry_reviewed"] = False
            stages["voices_ready"] = False
    if annotating is not None:
        stages["annotating"] = bool(annotating)
    if annotated is not None:
        stages["annotated"] = bool(annotated)
        stages["annotating"] = False
        stages["registry_reviewed"] = False
        stages["voices_ready"] = False
    if registry_reviewed is not None:
        stages["registry_reviewed"] = bool(registry_reviewed)
        if not registry_reviewed:
            stages["voices_ready"] = False
    if voices_ready is not None:
        stages["voices_ready"] = bool(voices_ready)
        if voices_ready:
            stages["registry_reviewed"] = True
    manifest["stages"] = stages
    write_json_atomic(book_root / BOOK_MANIFEST, manifest)


def _update_book_last_read(book_root: Path, chapter: str, unit_id: int) -> None:
    manifest = _read_book_manifest(book_root)
    if not manifest:
        manifest = {
            "schema": "readalong_book_v1",
            "title": book_root.name,
            "slug": book_root.name,
            "stages": _infer_stages_from_artifacts(book_root),
        }
    manifest["last_read"] = {"chapter": chapter, "unit_id": int(unit_id)}
    write_json_atomic(book_root / BOOK_MANIFEST, manifest)


def _infer_stages_from_artifacts(book_root: Path) -> Dict[str, bool]:
    chapter_count = _glob_count(book_root / "chapters", "*.txt")
    annotation_count, unit_count = _read_along_readiness_counts(book_root)
    registry = _read_json_if_exists(book_root / "registry.json")
    voice_count, voice_total = _registry_voice_readiness(book_root, registry)
    return {
        "initialized": chapter_count > 0,
        "global_registry": (book_root / "registry.json").exists(),
        "annotated": chapter_count > 0 and annotation_count >= chapter_count and unit_count >= chapter_count,
        "registry_reviewed": voice_count >= voice_total,
        "voices_ready": chapter_count > 0 and unit_count >= chapter_count and voice_count >= voice_total,
    }


def _registry_voice_readiness(book_root: Path, registry: Dict[str, Any]) -> Tuple[int, int]:
    characters = registry.get("characters", {}) if isinstance(registry.get("characters"), dict) else {}
    ready = 0
    total = 0
    for role_id, record in characters.items():
        if not isinstance(record, dict):
            continue
        total += 1
        voice_path = str(record.get("voice_config_path") or "").strip()
        current_hash = voice_profile_hash(record)
        cached_hash = str(record.get("voice_config_hash") or "")
        sample_path = book_root / "voices" / "_samples" / f"{role_id}.wav"
        if voice_path and (book_root / voice_path).exists() and cached_hash == current_hash and sample_path.exists():
            ready += 1
    return ready, total


def _selected_last_read_unit_id(book_root: Path, chapter: str, units: List[Dict[str, Any]]) -> Optional[int]:
    last_read = _manifest_last_read(_read_book_manifest(book_root))
    if str(last_read.get("chapter", "")) != str(chapter):
        return None
    try:
        unit_id = int(last_read.get("unit_id"))
    except (TypeError, ValueError):
        return None
    valid_unit_ids = {int(unit.get("unit_id")) for unit in units if unit.get("unit_id") is not None}
    return unit_id if unit_id in valid_unit_ids else None


def _validate_reading_position(book_root: Path, chapter: str, unit_id: int) -> None:
    chapter = str(chapter).strip()
    if not chapter:
        raise ValueError("chapter is required")
    unit_id = int(unit_id)
    paths = BookPaths(book_root)
    units_path = paths.read_along_units(chapter)
    if not units_path.exists():
        raise ValueError("Read-along units are required before saving a reading position.")
    units = read_json(units_path).get("units", [])
    valid_unit_ids = {int(unit.get("unit_id")) for unit in units if unit.get("unit_id") is not None}
    if unit_id not in valid_unit_ids:
        raise ValueError(f"Read-along unit not found: {unit_id}")


def _manifest_last_read(manifest: Dict[str, Any]) -> Dict[str, Any]:
    last_read = manifest.get("last_read", {}) if manifest else {}
    return dict(last_read) if isinstance(last_read, dict) else {}


def _last_read_label(last_read: Dict[str, Any]) -> str:
    chapter = str(last_read.get("chapter", "")).strip()
    if not chapter:
        return "Not started"
    unit_id = last_read.get("unit_id")
    if unit_id is None:
        return chapter
    return f"{chapter}, segment {int(unit_id) + 1}"


def _safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-_")
    slug = slug[:max_length].strip("-_")
    return slug or "book"


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _glob_count(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob(pattern))


def _read_along_readiness_counts(book_root: Path) -> Tuple[int, int]:
    paths = BookPaths(book_root)
    chapters_dir = book_root / "chapters"
    if not chapters_dir.exists():
        return 0, 0
    annotation_count = 0
    unit_count = 0
    for chapter_file in sorted(chapters_dir.glob("*.txt")):
        chapter = chapter_file.stem
        annotation_path = paths.annotation(chapter)
        if not annotation_path.exists():
            continue
        payload = _read_json_if_exists(annotation_path)
        if not _annotation_matches_chapter_quotes(paths, chapter, payload):
            continue
        annotation_count += 1
        if paths.read_along_units(chapter).exists():
            unit_count += 1
    return annotation_count, unit_count


def _recursive_glob_count(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def _same_path(left: Path, right: Optional[Path]) -> bool:
    if right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _multipart_value(form: cgi.FieldStorage, key: str) -> str:
    if key not in form:
        return ""
    field = form[key]
    if isinstance(field, list):
        field = field[0] if field else None
    if field is None:
        return ""
    return str(getattr(field, "value", "") or "")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Read Along</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #68707d;
      --line: #d7dce2;
      --paper: #fbfaf8;
      --panel: #f3f5f6;
      --accent: #c88731;
      --selected: #dce9f7;
      --current: #f6d56f;
      --buffered: #d8ead7;
      --book-grid-columns: minmax(160px, 2fr) minmax(110px, 0.9fr) minmax(72px, 0.45fr) minmax(96px, 0.6fr) minmax(118px, 0.75fr) minmax(70px, 0.45fr) minmax(120px, 0.8fr) 132px;
      --book-shell-columns: 32px minmax(0, 1fr) 32px;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: #e8eaed;
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    .library-view {
      min-height: 100vh;
      padding: 24px clamp(16px, 4vw, 48px);
      background: #f6f7f8;
    }
    .library-shell {
      max-width: 1120px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .library-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }
    .library-head h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
    }
    .library-root {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .book-list {
      display: grid;
      gap: 8px;
    }
    .book-list-header-shell,
    .book-row-shell {
      display: grid;
      grid-template-columns: var(--book-shell-columns);
      gap: 8px;
      align-items: center;
    }
    .book-list-header {
      display: grid;
      grid-template-columns: var(--book-grid-columns);
      gap: 12px;
      align-items: center;
      padding: 0 12px;
      border: 1px solid transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .book-row {
      display: grid;
      grid-template-columns: var(--book-grid-columns);
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(32, 33, 36, 0.06);
    }
    .delete-book,
    .share-book {
      width: 32px;
      min-width: 32px;
      padding: 0;
      min-height: 32px;
      border-color: transparent;
      background: transparent;
    }
    .delete-book {
      color: #9b3b32;
      font-weight: 700;
    }
    .delete-book:hover {
      border-color: #e1b7b2;
      background: #fff1ef;
    }
    .share-book {
      color: #4f5c69;
    }
    .share-book svg {
      display: block;
      width: 17px;
      height: 17px;
      margin: 0 auto;
      stroke: currentColor;
    }
    .share-book:hover:not(:disabled) {
      border-color: #cbd2da;
      background: #f7f8fa;
    }
    .book-cell {
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .book-title-cell {
      color: var(--ink);
      font-weight: 700;
      display: grid;
      gap: 2px;
      overflow: hidden;
    }
    .book-title-open {
      cursor: pointer;
    }
    .book-title-open:hover .book-title-line,
    .book-title-open:focus .book-title-line {
      text-decoration: underline;
    }
    .book-title-line {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .book-author-line {
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .book-status-cell {
      color: var(--ink);
      font-weight: 700;
    }
    .book-empty {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }
    .book-status {
      display: inline-block;
      margin-right: 8px;
      font-weight: 700;
      color: var(--ink);
    }
    .book-action-cell {
      display: grid;
      gap: 6px;
    }
    .book-action-cell button {
      width: 100%;
      min-height: 30px;
      padding-inline: 8px;
      font-size: 13px;
    }
    .book-secondary-action {
      border-color: #cbd2da;
      color: #4f5c69;
      background: #f7f8fa;
    }
    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      margin-left: 8px;
      border: 2px solid #cfd5dc;
      border-top-color: var(--accent);
      border-radius: 50%;
      vertical-align: -2px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .library-status {
      min-height: 24px;
      max-height: 76px;
      overflow: auto;
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 13px;
    }
    .add-book {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 0.8fr) minmax(0, 0.8fr) minmax(0, 0.6fr) max-content;
      gap: 10px;
      align-items: end;
      padding-top: 4px;
    }
    .add-book label {
      display: grid;
      align-items: stretch;
      min-width: 0;
    }
    .add-book input {
      width: 100%;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .add-book button {
      white-space: nowrap;
    }
    .import-package {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.7fr) max-content;
      gap: 10px;
      align-items: end;
      margin-top: 10px;
    }
    .import-package label {
      display: grid;
      min-width: 0;
    }
    .import-package input {
      width: 100%;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .import-package button {
      white-space: nowrap;
    }
    .registry-panel {
      display: grid;
      gap: 12px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .registry-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .registry-head h2 {
      margin: 0;
      font-size: 18px;
    }
    #registry-save-all.saved {
      border-color: #2f7b44;
      background: #3fa65c;
      color: #fff;
    }
    .registry-list {
      display: grid;
      gap: 10px;
    }
    .registry-card {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fafb;
    }
    .registry-card-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .registry-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }
    .registry-meta {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .registry-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
    }
    .registry-grid label {
      display: grid;
      align-items: stretch;
      min-width: 0;
    }
    .registry-grid input,
    .registry-grid select {
      width: 100%;
      min-width: 0;
    }
    .registry-actions {
      display: flex;
      justify-content: end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(220px, 280px) 1fr;
      min-height: 100vh;
    }
    .app.sidebar-hidden {
      grid-template-columns: 0 minmax(0, 1fr);
    }
    .app.sidebar-hidden .sidebar {
      width: 0;
      min-width: 0;
      overflow: hidden;
      border-right: 0;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }
    .chapter-tab {
      position: fixed;
      left: 280px;
      top: 50%;
      z-index: 30;
      min-width: 34px;
      min-height: 92px;
      padding: 10px 6px;
      writing-mode: vertical-rl;
      transform: translateY(-50%);
      border-radius: 0 8px 8px 0;
      border-left: 0;
      background: #fff;
      box-shadow: 0 8px 22px rgba(32, 33, 36, 0.16);
    }
    .app.sidebar-hidden .chapter-tab {
      left: 0;
    }
    .chapter-tab:active:not(:disabled) {
      transform: translateY(calc(-50% + 1px));
    }
    .brand {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      font-size: 18px;
      font-weight: 700;
    }
    .chapters {
      overflow: auto;
      padding: 8px;
    }
    .chapter {
      width: 100%;
      min-height: 36px;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      text-align: left;
      padding: 8px 10px;
      cursor: pointer;
    }
    .chapter:hover { background: #e3e7ea; }
    .chapter.active { background: #fff; box-shadow: inset 3px 0 0 var(--accent); }
    .reader {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto auto;
      min-width: 0;
      min-height: 100vh;
      position: relative;
    }
    .toolbar {
      display: grid;
      gap: 8px;
      align-items: start;
      padding: 12px 16px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    .session-settings-row,
    .session-controls-row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .session-controls-row {
      padding-top: 2px;
    }
    .generation-hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      max-width: 360px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
    }
    .narrator-control {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
    }
    #narrator-summary {
      max-width: 260px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    input, select, button {
      font: inherit;
      min-height: 32px;
    }
    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 4px 8px;
      color: var(--ink);
    }
    button {
      border: 1px solid #b8c0c8;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 5px 10px;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease;
      box-shadow: 0 1px 0 rgba(32, 33, 36, 0.08);
    }
    button:hover:not(:disabled) {
      border-color: #89939c;
      background: #f8fafb;
      box-shadow: 0 2px 6px rgba(32, 33, 36, 0.12);
    }
    button:active:not(:disabled) {
      transform: translateY(1px);
      box-shadow: inset 0 1px 2px rgba(32, 33, 36, 0.12);
    }
    button:focus-visible {
      outline: 2px solid rgba(181, 125, 45, 0.5);
      outline-offset: 2px;
    }
    button.primary {
      border-color: #936625;
      background: #b57d2d;
      color: white;
    }
    button.primary:hover:not(:disabled) {
      border-color: #7c5118;
      background: #a56f24;
    }
    button:disabled, input:disabled, select:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .page-shell {
      position: relative;
      min-height: calc(100vh - 190px);
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      padding: 24px clamp(12px, 3vw, 42px);
    }
    .page-wrap {
      overflow: hidden;
      padding: 0;
    }
    .page-spread {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      max-width: 920px;
      margin: 0 auto;
    }
    .app.sidebar-hidden .page-spread {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      max-width: 1320px;
    }
    .page-nav {
      width: 38px;
      min-height: 56px;
      font-size: 26px;
      line-height: 1;
    }
    .page {
      max-width: 820px;
      height: calc(100vh - 230px);
      min-height: 520px;
      margin: 0 auto;
      padding: clamp(26px, 4vw, 54px);
      background: var(--paper);
      box-shadow: 0 2px 20px rgba(32, 33, 36, 0.14);
      border: 1px solid rgba(0, 0, 0, 0.08);
      font-family: Georgia, "Times New Roman", serif;
      font-size: 20px;
      line-height: 1.72;
      white-space: pre-wrap;
      overflow: hidden;
      box-sizing: border-box;
    }
    .page-indicator {
      min-height: 26px;
      padding: 0 16px 8px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }
    .audiobook-app {
      display: grid;
      gap: 12px;
      align-content: start;
      min-height: 100vh;
      padding: 24px clamp(16px, 4vw, 48px);
      background: #f6f7f8;
      overflow: auto;
    }
    .audiobook-app[hidden] {
      display: none;
    }
    .audiobook-head,
    .audiobook-settings,
    .audiobook-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .audiobook-head h2 {
      margin: 0;
      font-size: 20px;
    }
    .audiobook-settings,
    .audiobook-actions {
      justify-content: flex-start;
    }
    .audiobook-inline {
      color: var(--ink);
    }
    .audiobook-chapters {
      display: grid;
      gap: 8px;
    }
    .audiobook-chapter {
      display: grid;
      grid-template-columns: minmax(0, 1fr) max-content max-content;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .audiobook-chapter-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }
    .audiobook-chapter-meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    #audiobook-player {
      width: 100%;
      margin-top: 6px;
    }
    .page-measurer {
      position: fixed;
      left: -10000px;
      top: 0;
      visibility: hidden;
      pointer-events: none;
    }
    .page-measurer .page {
      width: 100%;
      margin: 0;
    }
    .narrator-panel {
      margin: 10px 16px 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .narrator-panel[hidden] { display: none; }
    .unit {
      border-radius: 4px;
      outline: 1px dashed transparent;
      outline-offset: 2px;
      padding: 1px 0;
      cursor: pointer;
    }
    .unit.selected {
      background: transparent;
      outline: 1px dashed #d86b6b;
    }
    .unit.buffered { background: var(--buffered); }
    .unit.current { background: var(--current); }
    .tts-loading {
      position: fixed;
      inset: 0;
      z-index: 80;
      display: grid;
      place-items: center;
      background: rgba(20, 24, 28, 0.24);
    }
    .tts-loading[hidden] { display: none; }
    .tts-loading.paused .tts-spinner { display: none; }
    .session-error {
      margin: 8px 16px;
      padding: 10px 12px;
      border: 1px solid #d79090;
      background: #fff1f1;
      color: #7a1f1f;
      border-radius: 6px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .session-error[hidden] { display: none; }
    .return-prompt {
      position: absolute;
      inset: 0;
      z-index: 25;
      display: grid;
      place-items: center;
      background: rgba(20, 24, 28, 0.28);
    }
    .return-prompt[hidden] { display: none; }
    .return-prompt-panel {
      width: min(340px, calc(100vw - 40px));
      padding: 18px;
      border: 1px solid rgba(124, 91, 48, 0.24);
      border-radius: 8px;
      background: #fffaf0;
      box-shadow: 0 12px 34px rgba(32, 33, 36, 0.2);
    }
    .return-prompt-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .return-prompt-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .return-prompt-actions {
      display: flex;
      justify-content: end;
      gap: 8px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    .tts-loading-panel {
      display: inline-flex;
      align-items: center;
      gap: 12px;
      min-height: 46px;
      padding: 11px 16px;
      border: 1px solid rgba(124, 91, 48, 0.28);
      border-radius: 8px;
      background: rgba(255, 252, 245, 0.94);
      color: var(--ink);
      box-shadow: 0 8px 28px rgba(42, 36, 28, 0.16);
      font-size: 14px;
      font-weight: 600;
    }
    #tts-loading-stage {
      max-width: min(520px, calc(100vw - 72px));
      overflow-wrap: anywhere;
    }
    .tts-spinner {
      width: 18px;
      height: 18px;
      border: 3px solid rgba(181, 125, 45, 0.28);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: tts-spin 0.9s linear infinite;
    }
    @keyframes tts-spin {
      to { transform: rotate(360deg); }
    }
    .status {
      min-height: 38px;
      padding: 9px 16px;
      background: #fff;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 760px) {
      .library-head { align-items: start; flex-direction: column; }
      .book-list-header { display: none; }
      .book-row { grid-template-columns: auto minmax(0, 1fr); }
      .book-action-cell { grid-column: 1 / -1; }
      .add-book { grid-template-columns: 1fr; }
      .import-package { grid-template-columns: 1fr; }
      .app { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .chapters { display: flex; overflow-x: auto; padding: 8px; }
      .chapter { min-width: 190px; }
      .chapter-tab { left: 0; min-height: 82px; }
      .page-shell { grid-template-columns: minmax(0, 1fr); padding-inline: 10px; }
      .page-nav { display: none; }
      .page { font-size: 18px; box-shadow: none; }
    }
  </style>
</head>
<body>
  <section class="library-view" id="library-view" hidden>
    <div class="library-shell">
      <header class="library-head">
        <div>
          <h1>Read Along</h1>
          <div class="library-root" id="library-root"></div>
        </div>
        <button id="library-refresh">Refresh</button>
      </header>
      <section>
        <div class="add-book">
          <label>EPUB <input id="add-epub-file" type="file" accept=".epub,application/epub+zip"></label>
          <label>Title <input id="add-title" type="text" maxlength="120" placeholder="Book title"></label>
          <label>Author <input id="add-author" type="text" maxlength="120" placeholder="Author"></label>
          <label>Folder <input id="add-slug" type="text" maxlength="80" placeholder="book-folder"></label>
          <button class="primary" id="add-book">Add Book</button>
        </div>
        <div class="import-package">
          <label>ReadAlong Zip <input id="import-package-file" type="file" accept=".zip,.readalong.zip,application/zip"></label>
          <label>Folder <input id="import-package-slug" type="text" maxlength="80" placeholder="optional-folder"></label>
          <button id="import-package" type="button">Import Zip</button>
        </div>
      </section>
      <div class="library-status" id="library-status">Ready</div>
      <section>
        <div class="book-list" id="book-list"></div>
      </section>
      <section class="registry-panel" id="registry-panel" hidden>
        <div class="registry-head">
          <h2 id="registry-title">Registry Review</h2>
          <button id="registry-save-all">Save All</button>
          <button id="registry-close">Close</button>
        </div>
        <div class="registry-list" id="registry-list"></div>
      </section>
    </div>
  </section>
  <main class="app" id="reader-view">
    <button class="chapter-tab" id="toggle-sidebar" type="button" title="Show or hide chapters">Chapters</button>
    <aside class="sidebar">
      <div class="brand">Read Along</div>
      <nav class="chapters" id="chapters"></nav>
    </aside>
    <section class="reader">
      <div class="toolbar">
        <div class="session-settings-row">
          <label>Speed <input id="speed" type="number" min="1" max="4" step="0.05"></label>
          <label>Generation
            <select id="generation">
              <option value="balanced">Balanced (12-14 GB, RTF ~0.15)</option>
              <option value="fast">Burst (12-14 GB, fastest fill)</option>
              <option value="precise">Precise (12-14 GB, RTF ~0.25)</option>
            </select>
          </label>
          <div class="generation-hint" id="generation-hint">16 GB NVIDIA CUDA GPU recommended. Balanced uses the measured vLLM 12Hz seq2 profile. Smooth ceiling ~6.6x at 1.0x benchmark playback.</div>
          <input id="buffer" type="hidden" value="2">
          <label>Buffer s <input id="target-buffer" type="number" min="1" max="120" step="0.5"></label>
          <label>Chapter End
            <select id="chapter-end-behavior">
              <option value="stop">Stop at chapter end</option>
              <option value="continue">Continue to next chapter</option>
            </select>
          </label>
          <div class="narrator-control">
            <span id="narrator-summary">Narrator: loading</span>
            <button id="edit-narrator" type="button">Edit Narrator</button>
          </div>
        </div>
        <div class="session-controls-row">
          <button class="primary" id="start">Start Session</button>
          <button id="pause-session" type="button" disabled>Pause Session</button>
          <button id="end" disabled>End Session</button>
          <button id="open-audiobook" type="button">Audiobook</button>
        </div>
      </div>
      <section class="narrator-panel" id="narrator-panel" hidden>
        <div class="registry-head">
          <h2>Narrator Voice</h2>
          <button id="narrator-close" type="button">Close</button>
        </div>
        <div class="registry-grid">
          <label>Name <input id="narrator-display-name" data-narrator-field="display_name" type="text"></label>
          <label>Age <input id="narrator-age-stage" data-narrator-field="age_stage" type="text"></label>
          <label>Gender <input id="narrator-gender" data-narrator-field="gender" type="text"></label>
          <label>Personality <input id="narrator-personality" data-narrator-field="personality" type="text"></label>
          <label>Race / Ethnicity <select id="narrator-race" data-narrator-field="race_or_ethnicity"></select></label>
          <label>Accent <select id="narrator-accent" data-narrator-field="accent"></select></label>
          <label>Occupation <input id="narrator-occupation" data-narrator-field="occupation" type="text"></label>
        </div>
        <div class="registry-actions">
          <button id="save-narrator" type="button">Save Narrator</button>
        </div>
      </section>
      <div class="page-shell" id="page-shell">
        <button class="page-nav page-nav-prev" id="page-prev" type="button" aria-label="Previous page">&lt;</button>
        <div class="page-wrap" id="page-wrap">
          <div class="page-spread" id="page-spread"></div>
          <div id="reader-text" hidden></div>
        </div>
        <button class="page-nav page-nav-next" id="page-next" type="button" aria-label="Next page">&gt;</button>
      </div>
      <div class="page-indicator" id="page-indicator">Page 1</div>
      <div class="page-measurer" id="page-measurer" aria-hidden="true"></div>
      <div class="tts-loading" id="tts-loading-overlay" hidden aria-live="polite" aria-busy="true">
        <div class="tts-loading-panel">
          <span class="tts-spinner" aria-hidden="true"></span>
          <span id="tts-loading-stage">TTS stack loading</span>
          <button id="tts-loading-resume" type="button" hidden>Resume Session</button>
        </div>
      </div>
      <div class="session-error" id="session-error" hidden></div>
      <div class="return-prompt" id="return-prompt" hidden role="dialog" aria-modal="true" aria-labelledby="return-prompt-title">
        <div class="return-prompt-panel">
          <div class="return-prompt-title" id="return-prompt-title">Return to library?</div>
          <div class="return-prompt-copy" id="return-prompt-copy">The read-along session will pause while this prompt is open.</div>
          <div class="return-prompt-actions">
            <button id="return-prompt-resume" type="button">Resume</button>
            <button class="primary" id="return-prompt-yes" type="button">Yes, return</button>
          </div>
        </div>
      </div>
      <div class="status" id="status">Ready</div>
      <audio id="audio"></audio>
    </section>
  </main>
  <main class="audiobook-app" id="audiobook-view" hidden>
    <div class="audiobook-head">
      <h2>Audiobook</h2>
      <button id="audiobook-back" type="button">Book</button>
    </div>
    <div class="audiobook-settings">
      <label>Generation
        <select id="audiobook-generation">
          <option value="balanced">12Hz Balanced (12-14 GB, RTF ~0.15)</option>
          <option value="fast">12Hz Larger Windows (12-14 GB, chapter throughput)</option>
          <option value="precise">12Hz Smaller Windows (12-14 GB, debug)</option>
        </select>
      </label>
      <label>Player Speed <input id="audiobook-speed" type="number" min="0.5" max="4" step="0.05" value="1"></label>
      <label class="audiobook-inline"><input id="audiobook-auto-continue" type="checkbox" checked> Continue chapters</label>
      <div class="narrator-control">
        <span id="audiobook-narrator-summary">Audiobook narrator: loading</span>
        <button id="audiobook-edit-narrator" type="button">Edit Audiobook Narrator</button>
      </div>
    </div>
    <section class="narrator-panel" id="audiobook-narrator-panel" hidden>
      <div class="registry-head">
        <h2>Audiobook Narrator Voice</h2>
        <button id="audiobook-narrator-close" type="button">Close</button>
      </div>
      <div class="registry-grid">
        <label>Name <input id="audiobook-narrator-display-name" data-audiobook-narrator-field="display_name" type="text"></label>
        <label>Age <input id="audiobook-narrator-age-stage" data-audiobook-narrator-field="age_stage" type="text"></label>
        <label>Gender <input id="audiobook-narrator-gender" data-audiobook-narrator-field="gender" type="text"></label>
        <label>Personality <input id="audiobook-narrator-personality" data-audiobook-narrator-field="personality" type="text"></label>
        <label>Race / Ethnicity <select id="audiobook-narrator-race" data-audiobook-narrator-field="race_or_ethnicity"></select></label>
        <label>Accent <select id="audiobook-narrator-accent" data-audiobook-narrator-field="accent"></select></label>
        <label>Occupation <input id="audiobook-narrator-occupation" data-audiobook-narrator-field="occupation" type="text"></label>
      </div>
      <div class="registry-actions">
        <button id="audiobook-save-narrator" type="button">Save Audiobook Narrator</button>
      </div>
    </section>
    <div class="audiobook-actions">
      <button class="primary" id="audiobook-generate" type="button">Generate Selected</button>
      <button id="audiobook-regenerate" type="button">Regenerate Selected</button>
      <button id="audiobook-select-all" type="button">Select All</button>
    </div>
    <div class="audiobook-chapters" id="audiobook-chapters"></div>
    <audio id="audiobook-player" controls></audio>
  </main>
  <script>
    window.readAlongApp = true;
    const state = {
      chapters: [],
      chapter: "",
      text: "",
      units: [],
      selectedUnitId: 0,
      currentUnitId: null,
      pages: [],
      unitPage: {},
      pageIndex: 0,
      sidebarOpen: true,
      ready: [],
      sessionId: null,
      books: [],
      activeBook: null,
      registryBook: null,
      registryReview: null,
      narratorProfile: null,
      activeJobs: {},
      busySlug: "",
      sessionActive: false,
      sessionPaused: false,
      sessionChapterEndBehavior: "stop",
      audiobook: null,
      audiobookNarratorProfile: null,
      registrySaveTimer: null
    };
    const ACTION_LABELS = {
      initialize: "Initialize Book",
      build_registry: "Build Registry",
      annotate: "Annotate Book",
      review_registry: "Review Voices",
      prepare_voices: "Generate Voices",
      open: "Open"
    };
    const els = {
      libraryView: document.getElementById("library-view"),
      readerView: document.getElementById("reader-view"),
      bookList: document.getElementById("book-list"),
      libraryRoot: document.getElementById("library-root"),
      libraryStatus: document.getElementById("library-status"),
      addEpubFile: document.getElementById("add-epub-file"),
      addTitle: document.getElementById("add-title"),
      addAuthor: document.getElementById("add-author"),
      addSlug: document.getElementById("add-slug"),
      addButton: document.getElementById("add-book"),
      importPackageFile: document.getElementById("import-package-file"),
      importPackageSlug: document.getElementById("import-package-slug"),
      importPackageButton: document.getElementById("import-package"),
      registryPanel: document.getElementById("registry-panel"),
      registryTitle: document.getElementById("registry-title"),
      registryList: document.getElementById("registry-list"),
      registrySaveAll: document.getElementById("registry-save-all"),
      registryClose: document.getElementById("registry-close"),
      chapters: document.getElementById("chapters"),
      reader: document.getElementById("page-spread"),
      pageWrap: document.getElementById("page-wrap"),
      pageShell: document.getElementById("page-shell"),
      pageSpread: document.getElementById("page-spread"),
      pageIndicator: document.getElementById("page-indicator"),
      pageMeasurer: document.getElementById("page-measurer"),
      toggleSidebar: document.getElementById("toggle-sidebar"),
      pagePrev: document.getElementById("page-prev"),
      pageNext: document.getElementById("page-next"),
      status: document.getElementById("status"),
      speed: document.getElementById("speed"),
      generation: document.getElementById("generation"),
      generationHint: document.getElementById("generation-hint"),
      buffer: document.getElementById("buffer"),
      targetBuffer: document.getElementById("target-buffer"),
      chapterEndBehavior: document.getElementById("chapter-end-behavior"),
      narratorSummary: document.getElementById("narrator-summary"),
      editNarrator: document.getElementById("edit-narrator"),
      narratorPanel: document.getElementById("narrator-panel"),
      narratorClose: document.getElementById("narrator-close"),
      narratorRace: document.getElementById("narrator-race"),
      narratorAccent: document.getElementById("narrator-accent"),
      saveNarrator: document.getElementById("save-narrator"),
      start: document.getElementById("start"),
      pause: document.getElementById("pause-session"),
      end: document.getElementById("end"),
      openAudiobook: document.getElementById("open-audiobook"),
      audiobookView: document.getElementById("audiobook-view"),
      audiobookBack: document.getElementById("audiobook-back"),
      audiobookGeneration: document.getElementById("audiobook-generation"),
      audiobookSpeed: document.getElementById("audiobook-speed"),
      audiobookAutoContinue: document.getElementById("audiobook-auto-continue"),
      audiobookNarratorSummary: document.getElementById("audiobook-narrator-summary"),
      audiobookEditNarrator: document.getElementById("audiobook-edit-narrator"),
      audiobookNarratorPanel: document.getElementById("audiobook-narrator-panel"),
      audiobookNarratorClose: document.getElementById("audiobook-narrator-close"),
      audiobookNarratorRace: document.getElementById("audiobook-narrator-race"),
      audiobookNarratorAccent: document.getElementById("audiobook-narrator-accent"),
      audiobookSaveNarrator: document.getElementById("audiobook-save-narrator"),
      audiobookGenerate: document.getElementById("audiobook-generate"),
      audiobookRegenerate: document.getElementById("audiobook-regenerate"),
      audiobookSelectAll: document.getElementById("audiobook-select-all"),
      audiobookChapters: document.getElementById("audiobook-chapters"),
      audiobookPlayer: document.getElementById("audiobook-player"),
      audio: document.getElementById("audio"),
      ttsLoading: document.getElementById("tts-loading-overlay"),
      ttsLoadingStage: document.getElementById("tts-loading-stage"),
      ttsLoadingResume: document.getElementById("tts-loading-resume"),
      sessionError: document.getElementById("session-error"),
      returnPrompt: document.getElementById("return-prompt"),
      returnPromptYes: document.getElementById("return-prompt-yes"),
      returnPromptResume: document.getElementById("return-prompt-resume"),
      returnPromptCopy: document.getElementById("return-prompt-copy")
    };
    let sessionProgressTimer = null;
    let settingsSaveTimer = null;
    let topUpPromise = null;
    let preloadedReadAlongAudio = null;
    let preloadedReadAlongUrl = "";
    let returnPromptPreviousPaused = false;
    let audiobookPositionTimer = null;
    function compactStatusText(text, limit = 420) {
      const value = String(text || "");
      if (value.length <= limit) return value;
      return value.slice(0, Math.max(0, limit - 1)) + "...";
    }
    function setStatus(text) { els.status.textContent = text; }
    function showTtsLoading(visible, mode = "loading") {
      els.ttsLoading.hidden = !visible;
      els.ttsLoading.classList.toggle("paused", visible && mode === "paused");
      els.ttsLoadingResume.hidden = !(visible && mode === "paused");
      els.ttsLoading.setAttribute("aria-busy", visible && mode !== "paused" ? "true" : "false");
    }
    function setTtsLoadingStage(message) {
      els.ttsLoadingStage.textContent = String(message || "TTS stack loading");
    }
    function nextFrame() {
      return new Promise(resolve => requestAnimationFrame(() => resolve()));
    }
    function startSessionProgressPolling() {
      stopSessionProgressPolling();
      const poll = async () => {
        try {
          const payload = await api("/api/session/start-progress");
          if (payload.message) setTtsLoadingStage(payload.message);
        } catch (_error) {}
      };
      poll();
      sessionProgressTimer = window.setInterval(poll, 450);
    }
    function stopSessionProgressPolling() {
      if (sessionProgressTimer !== null) {
        window.clearInterval(sessionProgressTimer);
        sessionProgressTimer = null;
      }
    }
    function showSessionError(message) {
      const text = String(message || "");
      els.sessionError.hidden = !text;
      els.sessionError.textContent = text;
    }
    function setLibraryStatus(text) {
      const value = String(text || "");
      els.libraryStatus.textContent = compactStatusText(value);
      els.libraryStatus.title = value;
    }
    function setLibraryMode(enabled) {
      els.libraryView.hidden = !enabled;
      els.readerView.hidden = enabled;
      els.audiobookView.hidden = true;
    }
    function preferredViewMode() {
      try {
        return window.localStorage.getItem("readAlongViewMode") || "";
      } catch (_error) {
        return "";
      }
    }
    function setPreferredViewMode(mode) {
      try {
        window.localStorage.setItem("readAlongViewMode", mode);
      } catch (_error) {}
    }
    function resetReaderState() {
      state.chapter = "";
      state.text = "";
      state.units = [];
      state.selectedUnitId = 0;
      state.currentUnitId = null;
      state.pages = [];
      state.unitPage = {};
      state.pageIndex = 0;
      state.ready = [];
    }
    async function api(path, options = {}, requestOptions = {}) {
      const timeoutMs = Number(requestOptions.timeoutMs || 0);
      let timeoutId = null;
      let request = options;
      if (timeoutMs > 0) {
        const controller = new AbortController();
        timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
        request = { ...options, signal: controller.signal };
      }
      try {
        const response = await fetch(path, request);
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error || "Request failed");
        return payload;
      } catch (error) {
        if (error && error.name === "AbortError") {
          throw new Error("Request timed out after " + Math.round(timeoutMs / 1000) + "s.");
        }
        throw error;
      } finally {
        if (timeoutId !== null) window.clearTimeout(timeoutId);
      }
    }
    function settings() {
      return {
        playback_speed: Number(els.speed.value || 1),
        generation_mode: els.generation.value,
        buffer_limit: Number(els.buffer.value || 2),
        target_buffer_seconds: Number(els.targetBuffer.value || 20),
        max_buffer_seconds: Math.max(Number(els.targetBuffer.value || 20), Number(els.targetBuffer.value || 20) * 2),
        chapter_end_behavior: els.chapterEndBehavior.value
      };
    }
    async function loadLibrary(forceLibrary = false) {
      const payload = await api("/api/library");
      state.books = payload.books;
      state.activeBook = payload.active_book;
      renderLibrary(payload);
      setLibraryStatus(payload.books.length ? "Ready" : "No books found in this folder.");
      if (forceLibrary) setPreferredViewMode("library");
      if (!forceLibrary && payload.active_book && preferredViewMode() !== "library") {
        setLibraryMode(false);
        await loadState();
      } else {
        setLibraryMode(true);
      }
    }
    function renderLibrary(payload) {
      els.libraryRoot.textContent = payload.library_root;
      els.bookList.textContent = "";
      if (!payload.books.length) {
        const empty = document.createElement("div");
        empty.className = "book-empty";
        empty.textContent = "No books found in this folder.";
        els.bookList.appendChild(empty);
        return;
      }
      const headerShell = document.createElement("div");
      headerShell.className = "book-list-header-shell";
      headerShell.appendChild(document.createElement("div"));
      const header = document.createElement("div");
      header.className = "book-list-header";
      for (const label of ["Title", "Status", "Chapters", "Annotated", "Read-along", "Voices", "Last Read", ""]) {
        const cell = document.createElement("div");
        cell.textContent = label;
        header.appendChild(cell);
      }
      headerShell.appendChild(header);
      headerShell.appendChild(document.createElement("div"));
      els.bookList.appendChild(headerShell);
      for (const book of payload.books) {
        const activeJob = Object.values(state.activeJobs).find(job => job.slug === book.slug);
        const shell = document.createElement("div");
        shell.className = "book-row-shell";
        const row = document.createElement("div");
        row.className = "book-row";
        const deleteButton = document.createElement("button");
        deleteButton.className = "delete-book";
        deleteButton.type = "button";
        deleteButton.textContent = "X";
        deleteButton.title = "Delete book";
        deleteButton.setAttribute("aria-label", "Delete " + book.title);
        deleteButton.disabled = Boolean(state.busySlug);
        deleteButton.onclick = () => deleteBook(book);
        const share = document.createElement("button");
        share.className = "share-book";
        share.type = "button";
        share.title = book.open_enabled ? "Share Zip" : "Generate Voices before sharing";
        share.setAttribute("aria-label", "Share " + book.title);
        share.disabled = !book.open_enabled || Boolean(state.busySlug) || Boolean(activeJob);
        share.innerHTML = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7"/><path d="M12 16V4"/><path d="M7 9l5-5 5 5"/></svg>';
        share.onclick = () => shareBookPackage(book);
        const title = document.createElement("div");
        title.className = "book-cell book-title-cell";
        title.title = book.title;
        if (book.open_enabled) {
          title.classList.add("book-title-open");
          title.tabIndex = 0;
          title.setAttribute("role", "button");
          title.setAttribute("aria-label", "Open " + book.title);
          title.onclick = () => openBookFromTitle(book);
          title.onkeydown = event => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              openBookFromTitle(book);
            }
          };
        }
        const titleLine = document.createElement("div");
        titleLine.className = "book-title-line";
        titleLine.textContent = truncateBookTitle(book.title);
        title.appendChild(titleLine);
        if (book.author) {
          const authorLine = document.createElement("div");
          authorLine.className = "book-author-line";
          authorLine.textContent = book.author;
          authorLine.title = book.author;
          title.appendChild(authorLine);
        }
        if (state.busySlug === book.slug || activeJob) {
          const spinner = document.createElement("span");
          spinner.className = "spinner";
          spinner.setAttribute("aria-label", "Loading");
          titleLine.appendChild(spinner);
        }
        const status = bookCell("book-cell book-status-cell", activeJob ? jobStatusText(activeJob) : book.status_label);
        const chapters = bookCell("book-cell book-chapters-cell", String(book.chapter_count));
        const annotated = bookCell("book-cell book-annotated-cell", book.annotation_count + "/" + book.chapter_count);
        const units = bookCell("book-cell book-units-cell", book.read_along_unit_count + "/" + book.chapter_count);
        const voices = bookCell("book-cell book-voices-cell", voiceFractionText(book));
        const lastRead = bookCell("book-cell book-last-read-cell", book.last_read_label);
        const actions = document.createElement("div");
        actions.className = "book-action-cell";
        const button = document.createElement("button");
        button.className = "book-main-action";
        button.textContent = state.busySlug === book.slug ? pendingLabel(book.action_key) : (book.action_label || ACTION_LABELS[book.action_key] || "Open");
        button.disabled = Boolean(state.busySlug) || Boolean(activeJob);
        button.onclick = () => runBookAction(book);
        actions.appendChild(button);
        if (book.has_registry && book.action_key !== "review_registry" && book.status_key !== "fresh_added" && book.status_key !== "initialized") {
          const review = document.createElement("button");
          review.className = "book-secondary-action";
          review.type = "button";
          review.textContent = "Review Voices";
          review.title = "Review Voices";
          review.disabled = Boolean(state.busySlug) || Boolean(activeJob);
          review.onclick = () => openRegistryReview(book);
          actions.appendChild(review);
        }
        if (book.resume_annotation_enabled && book.action_key !== "annotate") {
          const resume = document.createElement("button");
          resume.className = "book-secondary-action";
          resume.type = "button";
          resume.textContent = book.resume_annotation_label || "Resume Annotation";
          resume.title = "Resume Annotation";
          resume.disabled = Boolean(state.busySlug) || Boolean(activeJob);
          resume.onclick = () => resumeAnnotation(book);
          actions.appendChild(resume);
        }
        row.appendChild(title);
        row.appendChild(status);
        row.appendChild(chapters);
        row.appendChild(annotated);
        row.appendChild(units);
        row.appendChild(voices);
        row.appendChild(lastRead);
        row.appendChild(actions);
        shell.appendChild(deleteButton);
        shell.appendChild(row);
        shell.appendChild(share);
        els.bookList.appendChild(shell);
      }
    }
    function bookCell(className, text) {
      const cell = document.createElement("div");
      cell.className = className;
      cell.textContent = text;
      return cell;
    }
    function voiceFractionText(book) {
      const ready = Number.isFinite(Number(book.voice_count)) ? Number(book.voice_count) : 0;
      const total = Number.isFinite(Number(book.voice_total)) ? Number(book.voice_total) : 0;
      return ready + "/" + total;
    }
    function truncateBookTitle(title) {
      if (title.length <= 40) return title;
      return title.slice(0, 37).trimEnd() + "...";
    }
    function pendingLabel(actionKey) {
      return {
        initialize: "Initializing...",
        build_registry: "Building registry...",
        annotate: "Annotating...",
        review_registry: "Loading registry...",
        prepare_voices: "Generating voices...",
        open: "Opening..."
      }[actionKey] || "Working...";
    }
    function actionStatusText(book) {
      return {
        initialize: "Initializing book...",
        build_registry: "Building registry...",
        annotate: "Annotating...",
        review_registry: "Loading registry...",
        prepare_voices: "Generating voices...",
        open: "Opening book..."
      }[book.action_key] || "Working...";
    }
    async function openBookFromTitle(book) {
      if (!book.open_enabled) return;
      await selectBook(book.slug);
    }
    async function resumeAnnotation(book) {
      state.busySlug = book.slug;
      setLibraryStatus("Resuming annotation...");
      renderLibrary({ library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      try {
        const payload = await api("/api/library/annotate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: book.slug })
        });
        state.books = payload.library.books;
        if (payload.job) {
          state.activeJobs[payload.job.job_id] = payload.job;
          setLibraryStatus(jobStatusText(payload.job));
          pollLibraryJob(payload.job.job_id).catch(error => setLibraryStatus(error.message));
        } else {
          setLibraryStatus(payload.book.title + ": " + payload.book.status_label + ".");
        }
      } catch (error) {
        setLibraryStatus(error.message);
      } finally {
        state.busySlug = "";
        renderLibrary({ library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      }
    }
    async function runBookAction(book) {
      if (book.action_key === "review_registry") {
        await openRegistryReview(book);
        return;
      }
      if (book.action_key === "open") {
        await selectBook(book.slug);
        return;
      }
      const endpoints = {
        initialize: "/api/library/initialize",
        build_registry: "/api/library/build-registry",
        annotate: "/api/library/annotate",
        prepare_voices: "/api/library/prepare-voices"
      };
      const endpoint = endpoints[book.action_key];
      if (!endpoint) return;
      state.busySlug = book.slug;
      setLibraryStatus(actionStatusText(book));
      renderLibrary({ library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      let latestLibrary = null;
      try {
        const payload = await api(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: book.slug })
        });
        state.books = payload.library.books;
        latestLibrary = payload.library;
        if (payload.job) {
          state.activeJobs[payload.job.job_id] = payload.job;
          setLibraryStatus(jobStatusText(payload.job));
          pollLibraryJob(payload.job.job_id).catch(error => setLibraryStatus(error.message));
        } else {
          setLibraryStatus(payload.book.title + ": " + payload.book.status_label + ".");
        }
      } catch (error) {
        setLibraryStatus(error.message);
      } finally {
        state.busySlug = "";
        renderLibrary(latestLibrary || { library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      }
    }
    function jobStatusText(job) {
      if (job.status === "failed") {
        return compactStatusText((job.failed_chapter ? job.failed_chapter + ": " : "") + job.error);
      }
      if (job.action_key === "audiobook") {
        if (job.status === "completed") {
          return "Audiobook audio ready.";
        }
        if (job.current_chapter) {
          return "Generating audiobook " + job.completed + "/" + job.total + ": " + job.current_chapter;
        }
        return "Generating audiobook " + job.completed + "/" + job.total + "...";
      }
      if (job.action_key === "prepare_voices") {
        if (job.status === "completed") {
          return "Voices ready.";
        }
        if (job.current_item) {
          return "Generating voice " + job.completed + "/" + job.total + ": " + job.current_item;
        }
        if (job.current_chapter) {
          return "Preparing chapter voices: " + job.current_chapter;
        }
        return "Generating voices " + job.completed + "/" + job.total + "...";
      }
      if (job.status === "completed") {
        return "Annotation complete.";
      }
      if (job.current_chapter) {
        return "Annotating chapter " + job.completed + "/" + job.total + ": " + job.current_chapter;
      }
      return "Annotating...";
    }
    async function pollLibraryJob(jobId) {
      while (state.activeJobs[jobId]) {
        const payload = await api("/api/library/job-status?job_id=" + encodeURIComponent(jobId));
        state.activeJobs[jobId] = payload.job;
        state.books = payload.library.books;
        setLibraryStatus(jobStatusText(payload.job));
        if (payload.job.status === "completed") {
          finishLibraryJob(jobId, payload);
          return payload;
        }
        if (payload.job.status === "failed") {
          finishLibraryJob(jobId, payload);
          return payload;
        }
        renderLibrary(payload.library);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    function finishLibraryJob(jobId, payload) {
      delete state.activeJobs[jobId];
      state.books = payload.library.books;
      state.activeBook = payload.library.active_book;
      renderLibrary(payload.library);
    }
    async function openRegistryReview(book) {
      state.registryBook = book;
      setLibraryStatus("Loading registry...");
      const payload = await api("/api/registry?slug=" + encodeURIComponent(book.slug));
      state.registryBook = payload.book;
      state.registryReview = payload.review;
      state.books = payload.library.books;
      renderLibrary(payload.library);
      renderRegistryPanel(payload.book, payload.review);
      setLibraryStatus(payload.book.title + ": Review Voices.");
    }
    function renderRegistryPanel(book, review) {
      state.registryBook = book;
      state.registryReview = review;
      els.registryPanel.hidden = false;
      els.registryTitle.textContent = truncateBookTitle(book.title || "Registry Review") + " - Registry Review";
      els.registryList.textContent = "";
      const entries = review && Array.isArray(review.entries) ? review.entries : [];
      for (const entry of entries) {
        const card = document.createElement("div");
        card.className = "registry-card";
        card.dataset.roleId = entry.role_id;
        const head = document.createElement("div");
        head.className = "registry-card-head";
        const title = document.createElement("div");
        title.className = "registry-title";
        title.textContent = entry.title || entry.role_id;
        title.title = entry.role_id;
        const sample = document.createElement("button");
        sample.type = "button";
        sample.textContent = "Sample";
        sample.onclick = () => playRegistrySample(entry);
        head.appendChild(title);
        head.appendChild(sample);
        card.appendChild(head);
        const meta = document.createElement("div");
        meta.className = "registry-meta";
        meta.textContent = (entry.kind || "character") + (entry.voice_config_path ? " | " + entry.voice_config_path : "");
        card.appendChild(meta);
        if (entry.editable) {
          const grid = document.createElement("div");
          grid.className = "registry-grid";
          for (const key of ["display_name", "age_stage", "gender", "personality", "race_or_ethnicity", "accent", "occupation", "aliases"]) {
            const field = registryField(entry, key, review);
            if (field) grid.appendChild(field);
          }
          card.appendChild(grid);
        }
        els.registryList.appendChild(card);
      }
    }
    function registryField(entry, key, review) {
      const fields = entry.fields || {};
      if (!(key in fields)) return null;
      const label = document.createElement("label");
      label.textContent = registryFieldLabel(key);
      let input;
      if (key === "accent" || key === "race_or_ethnicity") {
        input = document.createElement("select");
        const options = key === "accent" ? review.accent_options : review.race_or_ethnicity_options;
        const values = Array.isArray(options) ? [...options] : [""];
        if (!values.includes(fields[key])) values.push(fields[key]);
        for (const value of values) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value || "None";
          input.appendChild(option);
        }
        if (key === "accent") {
          const custom = document.createElement("input");
          custom.type = "text";
          custom.className = "custom-accent-input";
          custom.placeholder = "Custom accent";
          custom.dataset.customFieldKey = key;
          input.dataset.fieldKey = key;
          const showCustom = () => {
            custom.hidden = input.value !== "Custom";
            if (!custom.hidden) custom.focus();
          };
          input.onchange = showCustom;
          label.appendChild(input);
          label.appendChild(custom);
          input.value = fields[key] || "";
          showCustom();
          return label;
        }
      } else {
        input = document.createElement("input");
        input.type = "text";
      }
      input.value = fields[key] || "";
      input.dataset.fieldKey = key;
      label.appendChild(input);
      return label;
    }
    function registryFieldLabel(key) {
      return {
        display_name: "Name",
        age_stage: "Age",
        gender: "Gender",
        personality: "Personality",
        race_or_ethnicity: "Race / Ethnicity",
        accent: "Accent",
        occupation: "Occupation",
        aliases: "Aliases"
      }[key] || key;
    }
    async function saveRegistryCharacter(roleId) {
      if (!state.registryBook) return;
      const card = els.registryList.querySelector('[data-role-id="' + CSS.escape(roleId) + '"]');
      if (!card) return;
      const fields = {};
      for (const input of card.querySelectorAll("[data-field-key]")) {
        fields[input.dataset.fieldKey] = input.value;
      }
      setLibraryStatus("Saving registry and regenerating sample...");
      const payload = await api("/api/registry/save-character", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug: state.registryBook.slug, role_id: roleId, fields })
      });
      state.books = payload.library.books;
      renderLibrary(payload.library);
      renderRegistryPanel(payload.book, payload.review);
      setLibraryStatus("Saved " + roleId + ". Sample ready. Run Generate Voices again.");
    }
    async function saveRegistryAll() {
      if (!state.registryBook) return;
      const entries = [];
      for (const card of els.registryList.querySelectorAll("[data-role-id]")) {
        const fields = {};
        for (const input of card.querySelectorAll("[data-field-key]")) {
          fields[input.dataset.fieldKey] = registryInputValue(input, card);
        }
        entries.push({ role_id: card.dataset.roleId, fields });
      }
      setLibraryStatus("Saving registry changes...");
      resetRegistrySaveButton("Saving...");
      els.registrySaveAll.disabled = true;
      let saved = false;
      try {
        const payload = await api("/api/registry/save-characters", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: state.registryBook.slug, entries })
        });
        state.books = payload.library.books;
        renderLibrary(payload.library);
        renderRegistryPanel(payload.book, payload.review);
        if (payload.changed_role_ids.length > 0) {
          setLibraryStatus("Saved " + payload.changed_role_ids.length + " voice profiles. Run Generate Voices again.");
        } else {
          setLibraryStatus("No registry voice profile changes to save.");
        }
        saved = true;
      } finally {
        els.registrySaveAll.disabled = false;
        if (saved) markRegistrySaveSuccess();
        else resetRegistrySaveButton();
      }
    }
    function markRegistrySaveSuccess() {
      resetRegistrySaveButton("Saved");
      els.registrySaveAll.classList.add("saved");
      state.registrySaveTimer = window.setTimeout(() => resetRegistrySaveButton(), 1400);
    }
    function resetRegistrySaveButton(label = "Save All") {
      if (state.registrySaveTimer) {
        window.clearTimeout(state.registrySaveTimer);
        state.registrySaveTimer = null;
      }
      els.registrySaveAll.classList.remove("saved");
      els.registrySaveAll.textContent = label;
    }
    function registryInputValue(input, card) {
      if (input.dataset.fieldKey === "accent" && input.value === "Custom") {
        const custom = card.querySelector('[data-custom-field-key="accent"]');
        return custom ? custom.value : "";
      }
      return input.value;
    }
    async function playRegistrySample(entry) {
      if (!state.registryBook) return;
      if (!entry.sample_url) {
        setLibraryStatus("Sample missing. Run Generate Voices first.");
        return;
      }
      const url = new URL(entry.sample_url, window.location.origin);
      if (state.registryBook.slug && !url.searchParams.has("slug")) {
        url.searchParams.set("slug", state.registryBook.slug);
      }
      url.searchParams.set("t", String(Date.now()));
      const audio = new Audio(url.pathname + url.search);
      await audio.play();
      setLibraryStatus("Playing sample.");
    }
    async function deleteBook(book) {
      if (state.busySlug) return;
      if (!confirm('Delete "' + book.title + '"? This removes the book folder and generated files.')) return;
      state.busySlug = book.slug;
      setLibraryStatus("Deleting book...");
      renderLibrary({ library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      let latestLibrary = null;
      try {
        const payload = await api("/api/library/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slug: book.slug })
        });
        state.activeBook = payload.active_book;
        state.books = payload.library.books;
        latestLibrary = payload.library;
        setLibraryStatus("Deleted " + book.title + ".");
      } finally {
        state.busySlug = "";
        renderLibrary(latestLibrary || { library_root: els.libraryRoot.textContent, books: state.books, active_book: state.activeBook });
      }
    }
    async function selectBook(slug) {
      await api("/api/library/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug })
      });
      setPreferredViewMode("reader");
      resetReaderState();
      setLibraryMode(false);
      await loadState();
    }
    async function addBook() {
      const file = els.addEpubFile.files && els.addEpubFile.files[0];
      if (!file) throw new Error("Choose an EPUB file first.");
      const data = new FormData();
      data.append("epub", file, file.name);
      data.append("title", els.addTitle.value);
      data.append("author", els.addAuthor.value);
      data.append("slug", els.addSlug.value);
      els.addButton.disabled = true;
      setLibraryStatus("Uploading book...");
      try {
        const added = await api("/api/library/add-book", {
          method: "POST",
          body: data
        });
        state.activeBook = added.active_book;
        state.books = added.library.books;
        renderLibrary(added.library);
        setLibraryMode(true);
        setLibraryStatus("Added " + added.book.title + ". Use Initialize Book to extract chapters.");
      } finally {
        els.addButton.disabled = false;
      }
    }
    async function importPackage() {
      const file = els.importPackageFile.files && els.importPackageFile.files[0];
      if (!file) throw new Error("Choose a ReadAlong zip first.");
      const data = new FormData();
      data.append("package", file, file.name);
      data.append("slug", els.importPackageSlug.value);
      els.importPackageButton.disabled = true;
      setLibraryStatus("Importing ReadAlong package...");
      try {
        const imported = await api("/api/library/import-package", {
          method: "POST",
          body: data
        });
        state.activeBook = imported.active_book;
        state.books = imported.library.books;
        renderLibrary(imported.library);
        setLibraryMode(true);
        setLibraryStatus("Imported " + imported.book.title + ". Ready for read-along.");
      } finally {
        els.importPackageButton.disabled = false;
      }
    }
    function shareBookPackage(book) {
      setLibraryStatus("Preparing share zip for " + book.title + "...");
      window.location.href = "/api/library/export?slug=" + encodeURIComponent(book.slug);
    }
    function updateGenerationHint() {
      const hints = {
        balanced: "16 GB NVIDIA CUDA GPU recommended. Balanced uses the measured vLLM 12Hz seq2 profile. Smooth ceiling ~6.6x at 1.0x benchmark playback.",
        fast: "16 GB NVIDIA CUDA GPU recommended. Burst keeps the same resident VRAM profile but requests larger queue fills when buffer time allows.",
        precise: "16 GB NVIDIA CUDA GPU recommended. Precise generates one unit per call for debugging/fidelity; short-unit RTF was ~0.25 with ~3.9x smooth ceiling."
      };
      els.generationHint.textContent = hints[els.generation.value] || hints.balanced;
    }
    function slugFromFilename(name) {
      const stem = name.replace(/\.[^.]+$/, "");
      return (stem.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80).replace(/^-+|-+$/g, "")) || "book";
    }
    function titleFromFilename(name) {
      const stem = name.replace(/\.[^.]+$/, "");
      return (stem.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 120).trim()) || "Untitled Book";
    }
    function lockControls(locked) {
      for (const el of [els.speed, els.generation, els.buffer, els.targetBuffer, els.chapterEndBehavior, els.editNarrator, els.start]) {
        el.disabled = locked;
      }
      els.pause.disabled = !locked;
      els.pause.textContent = "Pause Session";
      els.end.disabled = !locked;
      state.sessionActive = locked;
      if (!locked) state.sessionPaused = false;
    }
    function populateSelect(select, options, value) {
      const selected = String(value || "");
      const values = Array.isArray(options) ? options : [""];
      const seen = new Set();
      select.textContent = "";
      for (const item of values.concat([selected])) {
        const optionValue = String(item || "");
        if (seen.has(optionValue)) continue;
        seen.add(optionValue);
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionValue || "Unspecified";
        select.appendChild(option);
      }
      select.value = selected;
    }
    async function loadNarratorProfile() {
      const payload = await api("/api/narrator-profile");
      state.narratorProfile = payload;
      els.narratorSummary.textContent = "Narrator: " + payload.summary;
      populateSelect(els.narratorRace, payload.race_or_ethnicity_options, payload.fields.race_or_ethnicity);
      populateSelect(els.narratorAccent, payload.accent_options, payload.fields.accent);
      for (const input of document.querySelectorAll("[data-narrator-field]")) {
        if (input.tagName.toLowerCase() === "select") continue;
        input.value = payload.fields[input.dataset.narratorField] || "";
      }
    }
    async function saveNarratorProfile() {
      const fields = {};
      for (const input of document.querySelectorAll("[data-narrator-field]")) {
        fields[input.dataset.narratorField] = input.value;
      }
      const payload = await api("/api/narrator-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields)
      });
      state.narratorProfile = payload;
      els.narratorSummary.textContent = "Narrator: " + payload.summary;
      setStatus("Narrator profile saved. It will be used when the next session starts.");
    }
    async function loadAudiobookNarratorProfile() {
      const payload = await api("/api/audiobook/narrator-profile");
      state.audiobookNarratorProfile = payload;
      els.audiobookNarratorSummary.textContent = "Audiobook narrator: " + payload.summary;
      populateSelect(els.audiobookNarratorRace, payload.race_or_ethnicity_options, payload.fields.race_or_ethnicity);
      populateSelect(els.audiobookNarratorAccent, payload.accent_options, payload.fields.accent);
      for (const input of document.querySelectorAll("[data-audiobook-narrator-field]")) {
        if (input.tagName.toLowerCase() === "select") continue;
        input.value = payload.fields[input.dataset.audiobookNarratorField] || "";
      }
    }
    async function saveAudiobookNarratorProfile() {
      const fields = {};
      for (const input of document.querySelectorAll("[data-audiobook-narrator-field]")) {
        fields[input.dataset.audiobookNarratorField] = input.value;
      }
      const payload = await api("/api/audiobook/narrator-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields)
      });
      state.audiobookNarratorProfile = payload;
      els.audiobookNarratorSummary.textContent = "Audiobook narrator: " + payload.summary;
      setStatus("Audiobook narrator saved. Regenerate audiobook chapters to use the new voice.");
    }
    async function loadState() {
      const payload = await api("/api/state");
      state.activeBook = payload.active_book;
      state.chapters = payload.chapters;
      els.speed.value = payload.settings.playback_speed;
      els.generation.value = payload.settings.generation_mode;
      els.buffer.value = payload.settings.buffer_limit;
      els.targetBuffer.value = payload.settings.target_buffer_seconds;
      els.chapterEndBehavior.value = payload.settings.chapter_end_behavior || "stop";
      updateGenerationHint();
      lockControls(Boolean(payload.session_active));
      await loadNarratorProfile();
      renderChapters();
      if (!state.chapter && state.chapters.length) {
        const lastRead = (payload.active_book && payload.active_book.last_read) || {};
        const desiredChapter = state.chapters.some(chapter => chapter.chapter === lastRead.chapter)
          ? lastRead.chapter
          : state.chapters[0].chapter;
        await loadChapter(desiredChapter);
      }
    }
    function renderChapters() {
      els.chapters.textContent = "";
      for (const chapter of state.chapters) {
        const button = document.createElement("button");
        button.className = "chapter" + (chapter.chapter === state.chapter ? " active" : "");
        button.textContent = String(chapter.index).padStart(3, "0") + " - " + chapter.title;
        button.onclick = () => loadChapter(chapter.chapter);
        els.chapters.appendChild(button);
      }
    }
    function nextChapterAfter(chapter) {
      const index = state.chapters.findIndex(item => item.chapter === chapter);
      if (index < 0 || index + 1 >= state.chapters.length) return null;
      return state.chapters[index + 1];
    }
    async function loadChapter(chapter, options = {}) {
      if (state.sessionActive && !options.allowDuringSession) return;
      setStatus("Loading chapter...");
      const payload = await api("/api/chapter/" + encodeURIComponent(chapter));
      state.chapter = payload.chapter;
      state.text = payload.text;
      state.units = payload.units;
      state.selectedUnitId = options.selectFirstUnit && state.units.length
        ? state.units[0].unit_id
        : payload.selected_unit_id ?? (state.units.length ? state.units[0].unit_id : 0);
      state.currentUnitId = null;
      state.pageIndex = 0;
      renderChapters();
      renderText();
      highlight();
      if (!state.sessionActive) els.start.disabled = !payload.units_ready;
      setStatus(payload.message || (payload.chapter + ": " + state.units.length + " units"));
    }
    function visiblePageCount() {
      return state.sidebarOpen ? 1 : 2;
    }
    function ensureAnchorPage(unitId = null) {
      const anchor = unitId ?? state.currentUnitId ?? state.selectedUnitId;
      const page = state.unitPage[String(anchor)];
      if (Number.isFinite(Number(page))) {
        state.pageIndex = Math.max(0, Math.min(Number(page), Math.max(0, state.pages.length - 1)));
        if (!state.sidebarOpen && state.pageIndex % 2 === 1) state.pageIndex -= 1;
      }
    }
    function turnPage(delta) {
      if (state.sessionActive) return;
      const step = visiblePageCount();
      state.pageIndex = Math.max(0, Math.min(Math.max(0, state.pages.length - 1), state.pageIndex + delta * step));
      if (!state.sidebarOpen && state.pageIndex % 2 === 1) state.pageIndex -= 1;
      renderPages();
    }
    function renderText() {
      paginateChapter();
      ensureAnchorPage();
      renderPages();
    }
    function paginateChapter() {
      const pageCount = visiblePageCount();
      const wrapWidth = els.pageWrap.clientWidth || 920;
      const pageWidth = Math.max(320, Math.floor(wrapWidth / pageCount) - 24);
      els.pageMeasurer.style.width = pageWidth + "px";
      state.pages = [];
      state.unitPage = {};
      els.pageMeasurer.textContent = "";
      let current = [];
      let page = createPageArticle();
      els.pageMeasurer.appendChild(page);
      let cursor = 0;
      const units = [...state.units].sort((a, b) => a.source_start - b.source_start);
      if (!units.length) {
        state.pages = [[{ before: state.text, unit: null }]];
        state.pageIndex = 0;
        return;
      }
      for (const unit of units) {
        const fragment = {
          before: state.text.slice(cursor, unit.source_start),
          unit
        };
        appendMeasuredFragment(page, fragment);
        const overflowed = page.clientHeight && page.scrollHeight > page.clientHeight + 1;
        if (overflowed && current.length) {
          state.pages.push(current);
          current = [];
          page = createPageArticle();
          els.pageMeasurer.textContent = "";
          els.pageMeasurer.appendChild(page);
          appendMeasuredFragment(page, fragment);
        }
        state.unitPage[String(unit.unit_id)] = state.pages.length;
        current.push(fragment);
        cursor = unit.source_end;
      }
      const tail = state.text.slice(cursor);
      if (tail || !current.length) current.push({ before: tail, unit: null });
      state.pages.push(current);
      state.pageIndex = Math.max(0, Math.min(state.pageIndex, Math.max(0, state.pages.length - 1)));
    }
    function createPageArticle() {
      const page = document.createElement("article");
      page.className = "page";
      return page;
    }
    function appendMeasuredFragment(page, fragment) {
      if (fragment.before) page.append(document.createTextNode(fragment.before));
      if (!fragment.unit) return;
      const span = document.createElement("span");
      span.className = "unit";
      span.textContent = unitText(fragment.unit);
      page.appendChild(span);
    }
    function unitText(unit) {
      return state.text.slice(unit.source_start, unit.source_end);
    }
    function renderPages() {
      els.pageSpread.textContent = "";
      const count = visiblePageCount();
      const total = Math.max(1, state.pages.length);
      state.pageIndex = Math.max(0, Math.min(state.pageIndex, total - 1));
      if (!state.sidebarOpen && state.pageIndex % 2 === 1) state.pageIndex -= 1;
      const end = Math.min(total, state.pageIndex + count);
      for (let index = state.pageIndex; index < end; index += 1) {
        els.pageSpread.appendChild(renderPageArticle(state.pages[index] || []));
      }
      if (count === 2 && end > state.pageIndex + 1) {
        els.pageIndicator.textContent = "Pages " + (state.pageIndex + 1) + "-" + end + " of " + total;
      } else {
        els.pageIndicator.textContent = "Page " + (state.pageIndex + 1) + " of " + total;
      }
      els.pagePrev.disabled = state.sessionActive || state.pageIndex <= 0;
      els.pageNext.disabled = state.sessionActive || end >= total;
      applyHighlights();
    }
    function renderPageArticle(fragments) {
      const page = createPageArticle();
      for (const fragment of fragments) {
        if (fragment.before) page.append(document.createTextNode(fragment.before));
        const unit = fragment.unit;
        if (!unit) continue;
        const span = document.createElement("span");
        span.className = "unit";
        span.dataset.unitId = unit.unit_id;
        span.textContent = unitText(unit);
        span.onclick = async () => {
          if (state.sessionActive) return;
          state.selectedUnitId = unit.unit_id;
          state.currentUnitId = null;
          ensureAnchorPage(unit.unit_id);
          renderPages();
          setStatus("Selected " + (unit.unit_id + 1) + "/" + state.units.length + ": " + unit.role);
          await saveReadingPositionById(unit.unit_id);
        };
        page.appendChild(span);
      }
      return page;
    }
    function unitById(unitId) {
      return state.units.find(unit => Number(unit.unit_id) === Number(unitId)) || null;
    }
    function applyHighlights() {
      const buffered = new Set(state.ready.map(item => item.unit_id));
      for (const span of els.pageSpread.querySelectorAll(".unit")) {
        const unitId = Number(span.dataset.unitId);
        span.classList.toggle("selected", !state.sessionActive && unitId === state.selectedUnitId);
        span.classList.toggle("buffered", buffered.has(unitId));
        span.classList.toggle("current", state.currentUnitId !== null && unitId === state.currentUnitId);
      }
    }
    function highlight(current = null) {
      if (current !== null) state.currentUnitId = current;
      applyHighlights();
    }
    async function saveReadingPositionById(unitId, options = {}) {
      const unit = unitById(unitId);
      if (!unit) return;
      await saveReadingPosition(unit, options);
    }
    async function saveReadingPosition(unit, options = {}) {
      if (!state.activeBook || !state.chapter) return;
      try {
        const payload = await api("/api/reading-position", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slug: state.activeBook.slug,
            chapter: state.chapter,
            unit_id: unit.unit_id
          })
        });
        if (payload.library) {
          state.books = payload.library.books;
          state.activeBook = payload.library.active_book;
          renderLibrary(payload.library);
        }
      } catch (error) {
        if (!options.silent) {
          setStatus("Selected " + (unit.unit_id + 1) + "/" + state.units.length + ". Last read not saved: " + error.message);
        }
      }
    }
    async function saveSettings() {
      if (state.sessionActive) return;
      try {
        const payload = await api("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings())
        });
        els.speed.value = payload.settings.playback_speed;
        els.generation.value = payload.settings.generation_mode;
        els.buffer.value = payload.settings.buffer_limit;
        els.targetBuffer.value = payload.settings.target_buffer_seconds;
        els.chapterEndBehavior.value = payload.settings.chapter_end_behavior || "stop";
        updateGenerationHint();
        setStatus("Settings saved.");
      } catch (error) {
        setStatus(error.message);
      }
    }
    function scheduleSettingsSave() {
      if (state.sessionActive) return;
      if (settingsSaveTimer !== null) window.clearTimeout(settingsSaveTimer);
      settingsSaveTimer = window.setTimeout(() => {
        settingsSaveTimer = null;
        saveSettings().catch(error => setStatus(error.message));
      }, 350);
    }
    function setAudiobookMode(enabled) {
      els.audiobookView.hidden = !enabled;
      els.readerView.hidden = enabled;
      if (enabled) els.libraryView.hidden = true;
      els.narratorPanel.hidden = true;
      els.audiobookNarratorPanel.hidden = true;
      if (!enabled) {
        els.readerView.hidden = false;
        renderText();
      }
    }
    async function openAudiobookView() {
      if (state.sessionActive) {
        setStatus("End the read-along session before opening audiobook generation.");
        return;
      }
      if (!state.activeBook) return;
      setAudiobookMode(true);
      await loadAudiobook();
    }
    async function loadAudiobook() {
      if (!state.activeBook) return;
      setStatus("Loading audiobook status...");
      const [payload] = await Promise.all([
        api("/api/audiobook?slug=" + encodeURIComponent(state.activeBook.slug)),
        loadAudiobookNarratorProfile()
      ]);
      state.audiobook = payload;
      renderAudiobook();
      setStatus("Audiobook ready.");
    }
    function renderAudiobook() {
      const payload = state.audiobook || { chapters: [], settings: {} };
      const settings = payload.settings || {};
      els.audiobookGeneration.value = settings.generation_mode || "balanced";
      els.audiobookSpeed.value = settings.playback_speed || 1;
      els.audiobookAutoContinue.checked = settings.auto_continue !== false;
      els.audiobookChapters.textContent = "";
      for (const chapter of payload.chapters || []) {
        const row = document.createElement("div");
        row.className = "audiobook-chapter";
        row.dataset.chapter = chapter.chapter;
        const label = document.createElement("label");
        label.className = "audiobook-chapter-title";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = !chapter.audio_ready;
        checkbox.dataset.chapter = chapter.chapter;
        label.appendChild(checkbox);
        label.append(" " + String(chapter.index).padStart(3, "0") + " - " + chapter.title);
        const meta = document.createElement("div");
        meta.className = "audiobook-chapter-meta";
        meta.textContent = chapter.audio_ready
          ? ("Ready" + (chapter.duration_seconds ? " - " + formatSeconds(chapter.duration_seconds) : ""))
          : "Not generated";
        const play = document.createElement("button");
        play.type = "button";
        play.textContent = "Play";
        play.disabled = !chapter.audio_ready;
        play.onclick = () => playAudiobookChapter(chapter);
        row.appendChild(label);
        row.appendChild(meta);
        row.appendChild(play);
        els.audiobookChapters.appendChild(row);
      }
    }
    function selectedAudiobookChapters() {
      return Array.from(els.audiobookChapters.querySelectorAll('input[type="checkbox"]:checked'))
        .map(input => input.dataset.chapter)
        .filter(Boolean);
    }
    function audiobookSettings() {
      return {
        generation_mode: els.audiobookGeneration.value,
        model_profile: "12hz",
        playback_speed: Number(els.audiobookSpeed.value || 1),
        auto_continue: els.audiobookAutoContinue.checked
      };
    }
    async function generateAudiobook(force = false) {
      if (!state.activeBook) return;
      const chapters = selectedAudiobookChapters();
      if (!chapters.length) {
        setStatus("Select at least one audiobook chapter.");
        return;
      }
      for (const button of [els.audiobookGenerate, els.audiobookRegenerate]) button.disabled = true;
      setStatus(force ? "Regenerating audiobook..." : "Generating audiobook...");
      try {
        const payload = await api("/api/audiobook/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slug: state.activeBook.slug,
            chapters,
            force,
            settings: audiobookSettings()
          })
        });
        if (payload.job) {
          state.activeJobs[payload.job.job_id] = payload.job;
          setStatus(jobStatusText(payload.job));
          await pollLibraryJob(payload.job.job_id);
        }
        await loadAudiobook();
      } finally {
        for (const button of [els.audiobookGenerate, els.audiobookRegenerate]) button.disabled = false;
      }
    }
    async function playAudiobookChapter(chapter) {
      if (!chapter || !chapter.audio_url) return;
      els.audiobookPlayer.src = chapter.audio_url + (chapter.audio_url.includes("?") ? "&" : "?") + "t=" + Date.now();
      els.audiobookPlayer.dataset.chapter = chapter.chapter;
      els.audiobookPlayer.playbackRate = Number(els.audiobookSpeed.value || 1);
      await els.audiobookPlayer.play();
      setStatus("Playing " + chapter.title + ".");
    }
    function currentAudiobookChapter() {
      const chapterId = els.audiobookPlayer.dataset.chapter;
      const chapters = (state.audiobook && state.audiobook.chapters) || [];
      return chapters.find(chapter => chapter.chapter === chapterId) || null;
    }
    function saveAudiobookPositionSoon() {
      if (!state.activeBook || !els.audiobookPlayer.dataset.chapter) return;
      if (audiobookPositionTimer !== null) return;
      audiobookPositionTimer = window.setTimeout(() => {
        audiobookPositionTimer = null;
        api("/api/audiobook/position", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slug: state.activeBook.slug,
            chapter: els.audiobookPlayer.dataset.chapter,
            position_seconds: els.audiobookPlayer.currentTime || 0
          })
        }).catch(error => setStatus(error.message));
      }, 1200);
    }
    async function playNextAudiobookChapter() {
      if (!els.audiobookAutoContinue.checked || !state.audiobook) return;
      const current = currentAudiobookChapter();
      if (!current) return;
      const chapters = state.audiobook.chapters || [];
      const index = chapters.findIndex(chapter => chapter.chapter === current.chapter);
      const next = chapters.slice(index + 1).find(chapter => chapter.audio_ready);
      if (next) await playAudiobookChapter(next);
    }
    function formatSeconds(seconds) {
      const total = Math.max(0, Math.round(Number(seconds) || 0));
      const minutes = Math.floor(total / 60);
      const remainder = total % 60;
      return minutes + ":" + String(remainder).padStart(2, "0");
    }
    async function startSession() {
      if (!state.chapter) return;
      if (!state.units.length) {
        setStatus("Process Book before starting read-along for this chapter.");
        return;
      }
      lockControls(true);
      showTtsLoading(true);
      setTtsLoadingStage("Starting read-along session...");
      showSessionError("");
      startSessionProgressPolling();
      setStatus("TTS stack loading...");
      await nextFrame();
      try {
        const payload = await api("/api/session/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter: state.chapter,
            start_unit_id: state.selectedUnitId,
            settings: settings()
          })
        });
        state.units = payload.units;
        state.sessionId = payload.session_id;
        state.sessionChapterEndBehavior = payload.settings.chapter_end_behavior || "stop";
        state.ready = payload.ready;
        state.currentUnitId = null;
        renderText();
        stopSessionProgressPolling();
        showTtsLoading(false);
        setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s. Starting playback...");
        await playReady();
        topUpBuffer();
      } catch (error) {
        try {
          await api("/api/session/end", { method: "POST" });
        } catch (_cleanupError) {}
        stopSessionProgressPolling();
        showTtsLoading(false);
        state.ready = [];
        lockControls(false);
        highlight();
        showSessionError(error.message);
        setStatus(error.message);
      }
    }
    function clearPreloadedReadAlongAudio() {
      preloadedReadAlongAudio = null;
      preloadedReadAlongUrl = "";
    }
    function preloadNextReadyAudio() {
      const next = state.ready[1];
      if (!next || !next.audio_url) {
        clearPreloadedReadAlongAudio();
        return;
      }
      if (preloadedReadAlongUrl === next.audio_url) return;
      preloadedReadAlongAudio = new Audio(next.audio_url);
      preloadedReadAlongAudio.preload = "auto";
      preloadedReadAlongAudio.playbackRate = Number(els.speed.value || 1);
      preloadedReadAlongUrl = next.audio_url;
      try {
        preloadedReadAlongAudio.load();
      } catch (_error) {}
    }
    async function playReady(options = {}) {
      if (!state.sessionActive) return;
      const item = state.ready[0];
      if (!item) {
        await endSession("Reached the end of the chapter.");
        return;
      }
      state.selectedUnitId = item.unit_id;
      state.currentUnitId = item.unit_id;
      ensureAnchorPage(item.unit_id);
      renderPages();
      highlight(item.unit_id);
      setStatus("Playing " + (item.unit_id + 1) + "/" + state.units.length);
      saveReadingPositionById(item.unit_id, { silent: true }).catch(error => setStatus(error.message));
      els.audio.src = item.audio_url;
      els.audio.playbackRate = Number(els.speed.value || 1);
      if (!els.returnPrompt.hidden) {
        state.sessionPaused = true;
        setStatus("Session paused.");
        return;
      }
      await els.audio.play();
      preloadNextReadyAudio();
      if (!options.deferTopUp) topUpBuffer();
    }
    async function pauseSession() {
      if (!state.sessionActive || state.sessionPaused) return;
      els.audio.pause();
      state.sessionPaused = true;
      els.pause.textContent = "Resume Session";
      setTtsLoadingStage("Paused");
      showTtsLoading(true, "paused");
      setStatus("Session paused.");
    }
    async function resumeSession() {
      if (!state.sessionActive || !state.sessionPaused) return;
      state.sessionPaused = false;
      els.pause.textContent = "Pause Session";
      showTtsLoading(false);
      if (state.ready.length && !els.audio.src) {
        await playReady();
        return;
      }
      try {
        await els.audio.play();
        topUpBuffer();
        setStatus("Resumed.");
      } catch (error) {
        setStatus(error.message);
      }
    }
    async function togglePauseSession() {
      if (state.sessionPaused) {
        await resumeSession();
      } else {
        await pauseSession();
      }
    }
    async function topUpBuffer(options = {}) {
      if (!state.sessionActive) return null;
      if (topUpPromise) return topUpPromise;
      const sessionId = state.sessionId;
      let promise;
      promise = (async () => {
        try {
          const body = {};
          const excludeUnitId = options.excludeUnitId ?? state.currentUnitId;
          if (excludeUnitId !== null && excludeUnitId !== undefined) body.exclude_unit_id = excludeUnitId;
          const payload = await api("/api/session/top-up", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          }, {
            timeoutMs: Number(options.timeoutMs || 120000)
          });
          if (!state.sessionActive || state.sessionId !== sessionId) return payload;
          state.ready = payload.ready;
          applyHighlights();
          if (typeof payload.ready_playback_seconds === "number" && !options.silent) {
            setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s");
          }
          return payload;
        } catch (error) {
          if (state.sessionActive && state.sessionId === sessionId && !options.silent) {
            setStatus("Buffer top-up failed: " + error.message);
          }
          return null;
        } finally {
          if (topUpPromise === promise) topUpPromise = null;
        }
      })();
      topUpPromise = promise;
      return promise;
    }
    async function advanceServerAfterLocalHandoff(finishedUnitId) {
      const sessionId = state.sessionId;
      try {
        const payload = await api("/api/session/advance", { method: "POST" });
        if (!state.sessionActive || state.sessionId !== sessionId) return;
        const currentUnitId = state.currentUnitId;
        state.ready = payload.ready;
        if (currentUnitId !== null && currentUnitId !== undefined) {
          const currentIndex = state.ready.findIndex(item => Number(item.unit_id) === Number(currentUnitId));
          if (currentIndex > 0) state.ready = state.ready.slice(currentIndex);
        }
        applyHighlights();
        preloadNextReadyAudio();
        topUpBuffer({ excludeUnitId: currentUnitId });
      } catch (error) {
        if (state.sessionActive && state.sessionId === sessionId) {
          setStatus("Session advance sync failed after unit " + finishedUnitId + ": " + error.message);
        }
      }
    }
    async function continueToNextChapter() {
      const nextChapter = nextChapterAfter(state.chapter);
      if (!nextChapter) {
        await endSession("Reached the end of the book.");
        return;
      }
      showTtsLoading(true);
      setTtsLoadingStage("Continuing to next chapter...");
      setStatus("Continuing to next chapter...");
      stopSessionProgressPolling();
      try {
        if (topUpPromise) {
          setTtsLoadingStage("Finishing current chapter buffer work.");
          await topUpPromise;
        }
        topUpPromise = null;
        state.sessionId = null;
        state.ready = [];
        clearPreloadedReadAlongAudio();
        state.currentUnitId = null;
        setTtsLoadingStage("Loading next chapter text and read-along units.");
        await loadChapter(nextChapter.chapter, { allowDuringSession: true, selectFirstUnit: true });
        setTtsLoadingStage("Reusing TTS stack, preparing next chapter voices.");
        const startRequest = api("/api/session/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter: state.chapter,
            start_unit_id: state.selectedUnitId,
            settings: settings(),
            reuse_active_tts: true
          })
        });
        startSessionProgressPolling();
        const payload = await startRequest;
        state.units = payload.units;
        state.sessionId = payload.session_id;
        state.sessionChapterEndBehavior = payload.settings.chapter_end_behavior || "stop";
        state.ready = payload.ready;
        state.currentUnitId = null;
        renderText();
        stopSessionProgressPolling();
        showTtsLoading(false);
        setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s. Continuing playback...");
        await playReady();
        topUpBuffer();
      } catch (error) {
        const message = error.message;
        stopSessionProgressPolling();
        showTtsLoading(false);
        await endSession("Session ended.");
        showSessionError(message);
      }
    }
    async function advanceSession() {
      if (!state.sessionActive) return;
      try {
        const finishedUnitId = state.currentUnitId;
        if (state.ready.length) {
          const currentIndex = state.ready.findIndex(item => Number(item.unit_id) === Number(finishedUnitId));
          if (currentIndex >= 0) {
            state.ready.splice(0, currentIndex + 1);
          } else {
            state.ready.shift();
          }
        }
        if (state.ready.length) {
          await playReady({ deferTopUp: true });
          advanceServerAfterLocalHandoff(finishedUnitId);
          return;
        }
        const payload = await api("/api/session/advance", { method: "POST" });
        state.ready = payload.ready;
        if (typeof payload.ready_playback_seconds === "number") {
          setStatus("Buffered " + payload.ready_playback_seconds.toFixed(1) + "s");
        }
        if (!state.ready.length) {
          if (payload.has_more_units) {
            showTtsLoading(true);
            setTtsLoadingStage("Building next audio buffer...");
            const toppedUp = await topUpBuffer({ silent: true, timeoutMs: 120000 });
            showTtsLoading(false);
            if (toppedUp && toppedUp.ready && toppedUp.ready.length) {
              state.ready = toppedUp.ready;
              await playReady();
              return;
            }
            if (!toppedUp) {
              await endSession("Buffer top-up failed or timed out.");
              return;
            }
          }
          if (payload.ended && state.sessionChapterEndBehavior === "continue") {
            await continueToNextChapter();
            return;
          }
          await endSession(payload.ended ? "Reached the end of the chapter." : "Buffer unavailable.");
          return;
        }
        await playReady();
      } catch (error) {
        setStatus(error.message);
        await endSession("Session ended.");
      }
    }
    async function endSession(message = "Session ended.") {
      try {
        await api("/api/session/end", { method: "POST" });
      } catch (error) {
        setStatus(error.message);
      }
      els.audio.pause();
      els.audio.removeAttribute("src");
      stopSessionProgressPolling();
      showTtsLoading(false);
      showSessionError("");
      els.returnPrompt.hidden = true;
      state.sessionPaused = false;
      returnPromptPreviousPaused = false;
      state.sessionId = null;
      topUpPromise = null;
      clearPreloadedReadAlongAudio();
      state.ready = [];
      state.currentUnitId = null;
      lockControls(false);
      renderPages();
      setStatus(message);
    }
    function showReturnPrompt() {
      if (els.readerView.hidden) return;
      returnPromptPreviousPaused = state.sessionActive && state.sessionPaused;
      if (state.sessionActive && !els.audio.paused && !returnPromptPreviousPaused) {
        els.audio.pause();
        state.sessionPaused = true;
      } else if (!state.sessionActive) {
        state.sessionPaused = false;
      }
      showTtsLoading(false);
      els.returnPromptCopy.textContent = state.sessionActive
        ? "The read-along session is paused. Return will end the session."
        : "Return to the library view.";
      els.returnPrompt.hidden = false;
      els.returnPromptResume.focus();
    }
    async function resumeFromReturnPrompt() {
      els.returnPrompt.hidden = true;
      if (state.sessionActive && returnPromptPreviousPaused) {
        returnPromptPreviousPaused = false;
        state.sessionPaused = true;
        els.pause.textContent = "Resume Session";
        setTtsLoadingStage("Paused");
        showTtsLoading(true, "paused");
        setStatus("Session paused.");
        return;
      }
      returnPromptPreviousPaused = false;
      if (state.sessionActive && state.sessionPaused) {
        state.sessionPaused = false;
        els.pause.textContent = "Pause Session";
        showTtsLoading(false);
        try {
          await els.audio.play();
          topUpBuffer();
          setStatus("Resumed.");
        } catch (error) {
          setStatus(error.message);
        }
        return;
      }
      state.sessionPaused = false;
    }
    function toggleReturnPrompt() {
      if (!els.returnPrompt.hidden) {
        resumeFromReturnPrompt().catch(error => setStatus(error.message));
      } else {
        showReturnPrompt();
      }
    }
    async function returnToLibraryFromPrompt() {
      els.returnPrompt.hidden = true;
      state.sessionPaused = false;
      returnPromptPreviousPaused = false;
      if (state.sessionActive) {
        await endSession("Session ended.");
      }
      await loadLibrary(true);
    }
    document.getElementById("library-refresh").onclick = () => loadLibrary(true).catch(error => alert(error.message));
    document.getElementById("add-book").onclick = () => addBook().catch(error => setLibraryStatus(error.message));
    document.getElementById("import-package").onclick = () => importPackage().catch(error => setLibraryStatus(error.message));
    els.registryClose.onclick = () => { els.registryPanel.hidden = true; };
    els.registrySaveAll.onclick = () => saveRegistryAll().catch(error => setLibraryStatus(error.message));
    els.editNarrator.onclick = () => { els.narratorPanel.hidden = false; };
    els.narratorClose.onclick = () => { els.narratorPanel.hidden = true; };
    els.saveNarrator.onclick = () => saveNarratorProfile().catch(error => setStatus(error.message));
    els.audiobookEditNarrator.onclick = () => { els.audiobookNarratorPanel.hidden = false; };
    els.audiobookNarratorClose.onclick = () => { els.audiobookNarratorPanel.hidden = true; };
    els.audiobookSaveNarrator.onclick = () => saveAudiobookNarratorProfile().catch(error => setStatus(error.message));
    els.openAudiobook.onclick = () => openAudiobookView().catch(error => setStatus(error.message));
    els.audiobookBack.onclick = () => {
      els.audiobookPlayer.pause();
      setAudiobookMode(false);
      setStatus("Reader ready.");
    };
    els.audiobookGenerate.onclick = () => generateAudiobook(false).catch(error => setStatus(error.message));
    els.audiobookRegenerate.onclick = () => generateAudiobook(true).catch(error => setStatus(error.message));
    els.audiobookSelectAll.onclick = () => {
      for (const input of els.audiobookChapters.querySelectorAll('input[type="checkbox"]')) input.checked = true;
    };
    els.audiobookSpeed.onchange = () => {
      els.audiobookPlayer.playbackRate = Number(els.audiobookSpeed.value || 1);
    };
    els.audiobookPlayer.ontimeupdate = saveAudiobookPositionSoon;
    els.audiobookPlayer.onended = () => playNextAudiobookChapter().catch(error => setStatus(error.message));
    els.toggleSidebar.onclick = () => {
      state.sidebarOpen = !state.sidebarOpen;
      els.readerView.classList.toggle("sidebar-hidden", !state.sidebarOpen);
      renderText();
    };
    els.pagePrev.onclick = () => turnPage(-1);
    els.pageNext.onclick = () => turnPage(1);
    els.returnPromptResume.onclick = () => resumeFromReturnPrompt().catch(error => setStatus(error.message));
    els.returnPromptYes.onclick = () => returnToLibraryFromPrompt().catch(error => setStatus(error.message));
    window.addEventListener("keydown", event => {
      const tag = String(document.activeElement && document.activeElement.tagName || "").toLowerCase();
      if (event.key === "Escape" && !els.readerView.hidden) {
        event.preventDefault();
        toggleReturnPrompt();
        return;
      }
      if (state.sessionActive || ["input", "select", "button", "textarea"].includes(tag)) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        turnPage(-1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        turnPage(1);
      }
    });
    window.addEventListener("resize", () => {
      if (!state.chapter) return;
      renderText();
    });
    els.addEpubFile.onchange = () => {
      const file = els.addEpubFile.files && els.addEpubFile.files[0];
      if (!file) return;
      if (!els.addTitle.value) els.addTitle.value = titleFromFilename(file.name);
      if (!els.addSlug.value) els.addSlug.value = slugFromFilename(file.name);
    };
    els.importPackageFile.onchange = () => {
      const file = els.importPackageFile.files && els.importPackageFile.files[0];
      if (!file) return;
      if (!els.importPackageSlug.value) els.importPackageSlug.value = slugFromFilename(file.name.replace(/\.readalong$/i, ""));
    };
    for (const el of [els.speed, els.generation, els.buffer, els.targetBuffer, els.chapterEndBehavior]) {
      el.addEventListener("change", scheduleSettingsSave);
    }
    els.generation.addEventListener("change", updateGenerationHint);
    updateGenerationHint();
    els.start.onclick = startSession;
    els.pause.onclick = () => togglePauseSession().catch(error => setStatus(error.message));
    els.ttsLoadingResume.onclick = () => resumeSession().catch(error => setStatus(error.message));
    els.end.onclick = () => endSession();
    els.audio.onended = advanceSession;
    loadLibrary().catch(error => {
      setLibraryMode(true);
      els.bookList.textContent = error.message;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
