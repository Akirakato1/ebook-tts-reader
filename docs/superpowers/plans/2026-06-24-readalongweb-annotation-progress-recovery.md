# Readalongweb Annotation Progress Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long book annotation observable and recoverable: the UI must show per-chapter progress, clear loading state on failure, and report the exact failed chapter/error instead of spinning forever.

**Architecture:** Keep the existing `readalongweb` library lifecycle, but run whole-book annotation as a tracked background job. `ReadAlongWebState` starts and owns jobs, `PrototypeUiController.annotate_read_along_book()` reports per-chapter progress through a callback, and the browser polls `/api/library/job-status` to update the book row. Failures are logged with chapter context and surfaced to the user.

**Tech Stack:** Python `http.server`, `threading`, existing `PrototypeUiController`, existing `FailureLogger`, plain JavaScript DOM polling, pytest.

---

## File Structure

- Modify `src/ebook_tts_pipeline/ui/web_app.py`
  - Add in-memory background job state.
  - Add `/api/library/job-status`.
  - Change `/api/library/annotate` to start an annotation job and return immediately.
  - Add frontend polling and robust `try/catch/finally` rendering for book actions.

- Modify `src/ebook_tts_pipeline/ui/controller.py`
  - Add progress callback support to `annotate_read_along_book()`.
  - Wrap per-chapter failures with chapter/action context.
  - Write a per-chapter progress artifact to the book folder.

- Modify `src/ebook_tts_pipeline/annotation/quote_attribution.py`
  - Add optional `FailureLogger`.
  - Log quote attribution model/validation failures with prompt, chapter, quote IDs, and raw payload/error.

- Modify `src/ebook_tts_pipeline/ui/controller.py` factory wiring
  - Pass `FailureLogger` into `QuoteAttributionService`.

- Modify `tests/test_read_along_web_app.py`
  - Add background job polling tests.
  - Add failed annotation job test proving no stale busy state assumptions at API level.
  - Update the lifecycle test to poll the job to completion.

- Modify `tests/test_ui_controller.py`
  - Add callback/progress tests for `annotate_read_along_book()`.
  - Add failure-wrapping test for chapter-specific annotation failure.

- Modify `tests/test_quote_attribution.py`
  - Add failure logging test for quote attribution validation/model errors.

---

## Task 1: Add Controller Progress Callback and Chapter Failure Context

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_ui_controller.py`

- [ ] **Step 1: Write the failing progress callback test**

Add this fake pipeline helper near existing test fakes in `tests/test_ui_controller.py`:

```python
class FakeProgressPipeline(FakePipeline):
    def annotate_chapter(self, chapter, lock_registry=False):
        self.calls.append(("annotate", chapter, lock_registry))
        self.paths.annotation(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.annotation(chapter),
            {"schema": "quote_attribution_v1", "roles": [], "quotes": []},
        )
        return {"schema": "quote_attribution_v1", "roles": [], "quotes": []}

    def build_read_along_units(self, chapter):
        self.calls.append(("build_units", chapter))
        self.paths.read_along_units(chapter).parent.mkdir(parents=True, exist_ok=True)
        payload = {"chapter": chapter, "units": []}
        write_json_atomic(self.paths.read_along_units(chapter), payload)
        return payload["units"]
```

Add this factory:

```python
def fake_progress_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts, config.book_root))
        return FakeProgressPipeline(config, calls)

    return factory
```

Add this test:

```python
def test_controller_annotate_read_along_book_reports_per_chapter_progress(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    for chapter in ["chapter_001", "chapter_002"]:
        paths.chapter_text(chapter).parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text(chapter).write_text(f"{chapter} text.", encoding="utf-8")
        paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            paths.sentence_artifact(chapter),
            {
                "chapter": chapter,
                "source_path": f"chapters/{chapter}.txt",
                "segmenter": {"name": "test"},
                "sentences": [{"idx": 0, "text": f"{chapter} text."}],
            },
        )
    write_json_atomic(
        paths.root / "toc.json",
        {
            "chapters": [
                {"index": 1, "chapter": "chapter_001", "title": "One", "source": "chapter_001.txt"},
                {"index": 2, "chapter": "chapter_002", "title": "Two", "source": "chapter_002.txt"},
            ]
        },
    )
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    progress = []
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_progress_pipeline_factory(calls),
        fake_tts=True,
    )

    result = controller.annotate_read_along_book(progress_callback=progress.append)

    assert result == {"chapters": 2, "annotated": 2, "units_built": 2}
    assert progress == [
        {"chapter": "chapter_001", "index": 1, "total": 2, "status": "started"},
        {"chapter": "chapter_001", "index": 1, "total": 2, "status": "completed"},
        {"chapter": "chapter_002", "index": 2, "total": 2, "status": "started"},
        {"chapter": "chapter_002", "index": 2, "total": 2, "status": "completed"},
    ]
    progress_file = read_json(paths.root / "read_along" / "annotation_progress.json")
    assert progress_file["status"] == "completed"
    assert progress_file["completed"] == 2
    assert progress_file["total"] == 2
```

- [ ] **Step 2: Write the failing chapter failure test**

Add:

```python
class FailingChapterPipeline(FakeProgressPipeline):
    def annotate_chapter(self, chapter, lock_registry=False):
        if chapter == "chapter_002":
            raise RuntimeError("model timed out")
        return super().annotate_chapter(chapter, lock_registry=lock_registry)


def failing_chapter_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts, config.book_root))
        return FailingChapterPipeline(config, calls)

    return factory
```

Add:

```python
def test_controller_annotate_read_along_book_reports_failed_chapter(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    for chapter in ["chapter_001", "chapter_002"]:
        paths.chapter_text(chapter).parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text(chapter).write_text(f"{chapter} text.", encoding="utf-8")
        paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            paths.sentence_artifact(chapter),
            {
                "chapter": chapter,
                "source_path": f"chapters/{chapter}.txt",
                "segmenter": {"name": "test"},
                "sentences": [{"idx": 0, "text": f"{chapter} text."}],
            },
        )
    write_json_atomic(
        paths.root / "toc.json",
        {
            "chapters": [
                {"index": 1, "chapter": "chapter_001", "title": "One", "source": "chapter_001.txt"},
                {"index": 2, "chapter": "chapter_002", "title": "Two", "source": "chapter_002.txt"},
            ]
        },
    )
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    progress = []
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=failing_chapter_pipeline_factory(calls),
        fake_tts=True,
    )

    with pytest.raises(RuntimeError, match="Annotation failed at chapter_002: model timed out"):
        controller.annotate_read_along_book(progress_callback=progress.append)

    assert progress[-1] == {
        "chapter": "chapter_002",
        "index": 2,
        "total": 2,
        "status": "failed",
        "error": "model timed out",
    }
    progress_file = read_json(paths.root / "read_along" / "annotation_progress.json")
    assert progress_file["status"] == "failed"
    assert progress_file["failed_chapter"] == "chapter_002"
    assert progress_file["error"] == "model timed out"
```

- [ ] **Step 3: Run failing controller tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_annotate_read_along_book_reports_per_chapter_progress tests\test_ui_controller.py::test_controller_annotate_read_along_book_reports_failed_chapter -v
```

Expected: FAIL because `annotate_read_along_book()` does not accept `progress_callback`.

- [ ] **Step 4: Implement controller progress callback**

In `src/ebook_tts_pipeline/ui/controller.py`, change the signature:

```python
def annotate_read_along_book(
    self,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, int]:
```

Replace the method body with:

```python
    pipeline = self._pipeline(needs_llm=True)
    chapters = [row.chapter for row in self.chapter_rows()]
    annotated = 0
    units_built = 0
    self._write_annotation_progress(
        {
            "status": "running",
            "completed": 0,
            "total": len(chapters),
            "current_chapter": "",
            "failed_chapter": "",
            "error": "",
        }
    )
    for index, chapter in enumerate(chapters, start=1):
        started_event = {"chapter": chapter, "index": index, "total": len(chapters), "status": "started"}
        self._emit_annotation_progress(started_event, progress_callback)
        self._write_annotation_progress(
            {
                "status": "running",
                "completed": index - 1,
                "total": len(chapters),
                "current_chapter": chapter,
                "failed_chapter": "",
                "error": "",
            }
        )
        try:
            if not self.paths.sentence_artifact(chapter).exists():
                pipeline.segment_chapter(chapter)
            annotation_payload = read_json(self.paths.annotation(chapter)) if self.paths.annotation(chapter).exists() else {}
            if not _is_quote_annotation_payload(annotation_payload):
                pipeline.annotate_chapter(chapter, lock_registry=True)
                annotated += 1
            pipeline.build_read_along_units(chapter)
            units_built += 1
        except Exception as exc:
            error = str(exc)
            failed_event = {
                "chapter": chapter,
                "index": index,
                "total": len(chapters),
                "status": "failed",
                "error": error,
            }
            self._emit_annotation_progress(failed_event, progress_callback)
            self._write_annotation_progress(
                {
                    "status": "failed",
                    "completed": index - 1,
                    "total": len(chapters),
                    "current_chapter": chapter,
                    "failed_chapter": chapter,
                    "error": error,
                }
            )
            raise RuntimeError(f"Annotation failed at {chapter}: {error}") from exc
        completed_event = {"chapter": chapter, "index": index, "total": len(chapters), "status": "completed"}
        self._emit_annotation_progress(completed_event, progress_callback)
        self._write_annotation_progress(
            {
                "status": "running",
                "completed": index,
                "total": len(chapters),
                "current_chapter": chapter,
                "failed_chapter": "",
                "error": "",
            }
        )
    self._write_annotation_progress(
        {
            "status": "completed",
            "completed": len(chapters),
            "total": len(chapters),
            "current_chapter": "",
            "failed_chapter": "",
            "error": "",
        }
    )
    return {
        "chapters": len(chapters),
        "annotated": annotated,
        "units_built": units_built,
    }
```

Add these methods inside `PrototypeUiController`:

```python
def _emit_annotation_progress(
    self,
    event: Dict[str, Any],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    if progress_callback is not None:
        progress_callback(dict(event))


def _write_annotation_progress(self, payload: Dict[str, Any]) -> None:
    progress_path = self.paths.root / "read_along" / "annotation_progress.json"
    write_json_atomic(progress_path, payload)
```

- [ ] **Step 5: Verify controller tests pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py::test_controller_annotate_read_along_book_reports_per_chapter_progress tests\test_ui_controller.py::test_controller_annotate_read_along_book_reports_failed_chapter -v
```

Expected: PASS.

---

## Task 2: Add Quote Attribution Failure Logging

**Files:**
- Modify: `src/ebook_tts_pipeline/annotation/quote_attribution.py`
- Modify: `src/ebook_tts_pipeline/ui/controller.py`
- Test: `tests/test_quote_attribution.py`

- [ ] **Step 1: Write failing quote attribution logging test**

Add to `tests/test_quote_attribution.py`:

```python
from ebook_tts_pipeline.debug_logging import FailureLogger
```

Add:

```python
class BadQuoteClient:
    def complete_json(self, system_prompt, user_prompt):
        return {"roles": ["Narrator"], "quotes": []}
```

Add:

```python
def test_quote_attribution_service_logs_validation_failure(tmp_path):
    extraction = QuoteExtraction(
        text='"Hello."',
        quotes=[QuoteSpan(idx=1, quote_id="q001", start=0, end=8, text='"Hello."')],
        narrator_spans=[],
    )
    logger = FailureLogger(tmp_path / "failures", context={"book_root": "book"})
    service = QuoteAttributionService(BadQuoteClient(), failure_logger=logger)

    with pytest.raises(QuoteAttributionValidationError, match="missing quote assignments"):
        service.attribute_quotes("chapter_002", extraction, {"characters": {}})

    logs = list((tmp_path / "failures").glob("*.json"))
    assert len(logs) == 1
    payload = read_json(logs[0])
    assert payload["event_type"] == "quote_attribution_validation_failed"
    assert payload["context"]["chapter"] == "chapter_002"
    assert payload["details"]["quote_ids"] == ["q001"]
    assert "Chapter: chapter_002" in payload["details"]["user_prompt"]
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_quote_attribution.py::test_quote_attribution_service_logs_validation_failure -v
```

Expected: FAIL because `QuoteAttributionService` does not accept `failure_logger`.

- [ ] **Step 3: Implement failure logger in `QuoteAttributionService`**

In `src/ebook_tts_pipeline/annotation/quote_attribution.py`, add import:

```python
from ebook_tts_pipeline.debug_logging import FailureLogger
```

Change constructor:

```python
def __init__(self, client: Any, failure_logger: FailureLogger | None = None) -> None:
    self.client = client
    self.failure_logger = failure_logger
```

Replace `attribute_quotes()` with:

```python
def attribute_quotes(
    self,
    chapter: str,
    extraction: QuoteExtraction,
    registry: Dict[str, Any],
) -> QuoteAttributionResult:
    prompt = render_quote_attribution_prompt(chapter, extraction, registry)
    quote_ids = [quote.quote_id for quote in extraction.quotes]
    try:
        payload = self.client.complete_json(SYSTEM_PROMPT, prompt)
    except Exception as exc:
        self._log_failure(
            "quote_attribution_model_failed",
            chapter=chapter,
            prompt=prompt,
            quote_ids=quote_ids,
            payload=None,
            exc=exc,
        )
        raise
    try:
        result = QuoteAttributionResult.from_dict(payload)
        validate_quote_attribution(
            result,
            quote_indices=[quote.idx for quote in extraction.quotes],
            known_role_ids=set(_registry_role_ids(registry)),
        )
    except Exception as exc:
        self._log_failure(
            "quote_attribution_validation_failed",
            chapter=chapter,
            prompt=prompt,
            quote_ids=quote_ids,
            payload=payload,
            exc=exc,
        )
        raise
    return result
```

Add method:

```python
def _log_failure(
    self,
    event_type: str,
    chapter: str,
    prompt: str,
    quote_ids: List[str],
    payload: Any,
    exc: BaseException,
) -> None:
    if self.failure_logger is None:
        return
    details = {
        "chapter": chapter,
        "quote_ids": quote_ids,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": prompt,
        "payload": payload,
        "raw_model_text": getattr(exc, "raw_text", None),
    }
    self.failure_logger.with_context(chapter=chapter).write_failure(event_type, details, exc=exc)
```

- [ ] **Step 4: Wire logger in controller factory**

In `_default_pipeline_factory()` in `src/ebook_tts_pipeline/ui/controller.py`, replace:

```python
quote_attribution_service=QuoteAttributionService(quote_client) if needs_llm else None,
```

with:

```python
quote_attribution_service=(
    QuoteAttributionService(
        quote_client,
        failure_logger=FailureLogger(
            config.debug_log_root,
            context={"book_root": config.book_root},
        ),
    )
    if needs_llm
    else None
),
```

- [ ] **Step 5: Verify quote logging test passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_quote_attribution.py::test_quote_attribution_service_logs_validation_failure -v
```

Expected: PASS.

---

## Task 3: Add Background Job State and Job Status API

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write failing background annotate job test**

Add to `tests/test_read_along_web_app.py`:

```python
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
        assert final["book"]["status_key"] == "registry_review"
        assert final["book"]["action_key"] == "review_registry"
    finally:
        _stop_server(server, thread)
```

Add helper near existing HTTP helpers:

```python
def _wait_for_job(base_url: str, job_id: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    last = {}
    while time.time() < deadline:
        last = _get_json(base_url + "/api/library/job-status?job_id=" + job_id)
        if last["job"]["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {last}")
```

Add import:

```python
import time
```

- [ ] **Step 2: Write failing job failure test**

Add:

```python
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
```

Add fake extractor:

```python
class _TwoChapterExtractor:
    def extract(self, epub_path, paths: BookPaths) -> EpubExtractResult:
        paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text("chapter_001").write_text('One. "Welcome."', encoding="utf-8")
        paths.chapter_text("chapter_002").write_text('Two. "Stop."', encoding="utf-8")
        return EpubExtractResult(chapters=["chapter_001", "chapter_002"], sources=[str(epub_path), str(epub_path)])
```

Add fake pipeline:

```python
class _FailingSecondChapterPipeline(_FakeLifecyclePipeline):
    def annotate_chapter(self, chapter: str, lock_registry: bool = True):
        if chapter == "chapter_002":
            raise RuntimeError("model timed out")
        return super().annotate_chapter(chapter, lock_registry=lock_registry)


def _failing_second_chapter_pipeline_factory(config, needs_llm, fake_tts):
    return _FailingSecondChapterPipeline(config)
```

- [ ] **Step 3: Run failing job tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_annotate_book_starts_background_job_and_reports_progress tests\test_read_along_web_app.py::test_failed_annotate_job_reports_failed_chapter_and_keeps_book_closed -v
```

Expected: FAIL because `/api/library/job-status` does not exist and `/api/library/annotate` is synchronous.

- [ ] **Step 4: Implement job dataclass and storage**

In `src/ebook_tts_pipeline/ui/web_app.py`, add imports:

```python
import time
import uuid
from dataclasses import dataclass, field
```

Replace the existing dataclass import:

```python
from dataclasses import dataclass
```

with:

```python
from dataclasses import dataclass, field
```

Add after `UploadedBook`:

```python
@dataclass
class LibraryJob:
    job_id: str
    slug: str
    action_key: str
    status: str = "queued"
    current_chapter: str = ""
    completed: int = 0
    total: int = 0
    failed_chapter: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "slug": self.slug,
            "action_key": self.action_key,
            "status": self.status,
            "current_chapter": self.current_chapter,
            "completed": self.completed,
            "total": self.total,
            "failed_chapter": self.failed_chapter,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
```

In `ReadAlongWebState.__post_init__()`, add:

```python
self.jobs: Dict[str, LibraryJob] = {}
self.jobs_by_slug: Dict[str, str] = {}
```

- [ ] **Step 5: Implement background annotate job methods**

Add to `ReadAlongWebState`:

```python
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
        return {"ok": True, "book": summary.to_payload(), "job": job.to_payload(), "library": self.library_payload()}
```

Add:

```python
def _run_annotate_job(self, job_id: str) -> None:
    with self.lock:
        job = self.jobs[job_id]
        summary = self._require_book_summary(job.slug)
        controller = self._make_controller(summary.book_root)
        job.status = "running"

    def on_progress(event: Dict[str, Any]) -> None:
        with self.lock:
            current = self.jobs[job_id]
            current.total = int(event.get("total") or current.total or 0)
            current.current_chapter = str(event.get("chapter") or current.current_chapter)
            if event.get("status") == "completed":
                current.completed = max(current.completed, int(event.get("index") or current.completed))
            if event.get("status") == "failed":
                current.status = "failed"
                current.failed_chapter = str(event.get("chapter") or "")
                current.error = str(event.get("error") or "")

    try:
        result = controller.annotate_read_along_book(progress_callback=on_progress)
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
        return

    with self.lock:
        job = self.jobs[job_id]
        job.status = "completed"
        job.completed = int(result["chapters"])
        job.total = int(result["chapters"])
        job.current_chapter = ""
        job.finished_at = time.time()
        summary = self._require_book_summary(job.slug)
        _update_book_stage(summary.book_root, annotating=False, annotated=True)
```

Add:

```python
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
```

- [ ] **Step 6: Add `annotating` manifest stage**

Extend `_book_status()` manifest branch before `if not stages.get("annotated")`:

```python
if stages.get("annotating"):
    return "annotating", "Annotating"
```

Extend `_book_action()`:

```python
if status_key == "annotating":
    return "annotate", "Annotating", False
```

Extend `_write_book_manifest()` stages:

```python
"annotating": False,
```

Extend `_update_book_stage()` args:

```python
annotating: Optional[bool] = None,
```

Inside `_update_book_stage()`:

```python
if annotating is not None:
    stages["annotating"] = bool(annotating)
```

When `annotated is not None`, add:

```python
stages["annotating"] = False
```

- [ ] **Step 7: Add job-status route**

In `do_GET`, before `/api/state`:

```python
if path == "/api/library/job-status":
    job_id = parse_qs(parsed.query).get("job_id", [""])[0]
    self._send_json(app_state.job_status(job_id))
    return
```

- [ ] **Step 8: Verify job tests pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_annotate_book_starts_background_job_and_reports_progress tests\test_read_along_web_app.py::test_failed_annotate_job_reports_failed_chapter_and_keeps_book_closed -v
```

Expected: PASS.

---

## Task 4: Update Existing Lifecycle Test to Poll Annotation Job

**Files:**
- Modify: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Update lifecycle test**

In `test_book_lifecycle_requires_explicit_steps_before_open_and_tracks_last_read`, replace:

```python
annotated = _post_json(base_url + "/api/library/annotate", {"slug": "lifecycle-book"})
```

with:

```python
started_annotation = _post_json(base_url + "/api/library/annotate", {"slug": "lifecycle-book"})
annotated = _wait_for_job(base_url, started_annotation["job"]["job_id"])
```

Keep the existing assertions:

```python
assert annotated["book"]["status_key"] == "registry_review"
assert annotated["book"]["action_key"] == "review_registry"
assert annotated["book"]["action_label"] == "Review Voices"
assert annotated["book"]["open_enabled"] is False
```

- [ ] **Step 2: Run lifecycle test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_book_lifecycle_requires_explicit_steps_before_open_and_tracks_last_read -v
```

Expected: PASS.

---

## Task 5: Fix Frontend Stale Spinner and Add Polling UI

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Test: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Add shell assertions for robust JS failure handling and polling**

In `test_home_page_serves_clean_reader_shell`, add:

```python
assert "pollLibraryJob" in response
assert "/api/library/job-status?job_id=" in response
assert "state.activeJobs" in response
assert "latestLibrary ||" in response
assert "setLibraryStatus(error.message)" in response
assert "finally {" in response
assert "Annotating chapter" in response
```

- [ ] **Step 2: Run failing shell test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v
```

Expected: FAIL until JavaScript is updated.

- [ ] **Step 3: Add JS state for active jobs**

In `INDEX_HTML`, extend `state`:

```javascript
activeJobs: {},
```

- [ ] **Step 4: Update `runBookAction()` with catch/finally that always rerenders**

Replace the `try/finally` block in `runBookAction(book)` with:

```javascript
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
```

- [ ] **Step 5: Add polling helpers**

Add after `runBookAction(book)`:

```javascript
function jobStatusText(job) {
  if (job.status === "failed") {
    return (job.failed_chapter ? job.failed_chapter + ": " : "") + job.error;
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
    renderLibrary(payload.library);
    setLibraryStatus(jobStatusText(payload.job));
    if (payload.job.status === "completed") {
      delete state.activeJobs[jobId];
      return payload;
    }
    if (payload.job.status === "failed") {
      delete state.activeJobs[jobId];
      return payload;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
```

- [ ] **Step 6: Make `renderLibrary()` display job progress in the status cell**

In `renderLibrary(payload)`, before creating `status`:

```javascript
const activeJob = Object.values(state.activeJobs).find(job => job.slug === book.slug);
```

Replace:

```javascript
const status = bookCell("book-cell book-status-cell", book.status_label);
```

with:

```javascript
const status = bookCell("book-cell book-status-cell", activeJob ? jobStatusText(activeJob) : book.status_label);
```

- [ ] **Step 7: Verify shell test passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py::test_home_page_serves_clean_reader_shell -v
```

Expected: PASS.

---

## Task 6: Verification

**Files:**
- Verify all modified source and tests.

- [ ] **Step 1: Run focused web tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_read_along_web_app.py -v
```

Expected: all web tests PASS.

- [ ] **Step 2: Run focused controller and quote attribution tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_controller.py tests\test_quote_attribution.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 3: Compile changed Python files**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\ebook_tts_pipeline\ui\web_app.py src\ebook_tts_pipeline\ui\controller.py src\ebook_tts_pipeline\annotation\quote_attribution.py
```

Expected: exit code 0.

- [ ] **Step 4: Run full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all tests PASS.

---

## Manual Verification on False Witness

1. Start the app from the book library folder:

```powershell
readalongweb
```

2. In the UI, click `Annotate Book` on False Witness.

3. Expected UI behavior:
   - Button switches to a busy state briefly.
   - Book status changes to `Annotating`.
   - Library status shows text like `Annotating chapter 1/23: chapter_001`.
   - Counts update as files are written.
   - If a chapter fails, status shows `chapter_00N: <error>`, spinner stops, and the row remains actionable.

4. Expected filesystem behavior:
   - `books/false-witness/read_along/annotation_progress.json` updates after each chapter.
   - `books/false-witness/annotations/chapter_00N.annotation.json` appears per successful chapter.
   - If quote attribution fails, a JSON file appears under `logs/annotation_failures` with `event_type` starting with `quote_attribution_`.

---

## Self-Review

- Spec coverage: This plan fixes the observed stuck UI, adds live annotation progress, preserves the managed book lifecycle, and gives chapter-specific failure evidence.
- Placeholder scan: No `TBD`, `TODO`, or undefined endpoint names are used. All new endpoints and function names are defined before use.
- Type consistency: Job payload fields are consistently `job_id`, `slug`, `action_key`, `status`, `current_chapter`, `completed`, `total`, `failed_chapter`, and `error`.
