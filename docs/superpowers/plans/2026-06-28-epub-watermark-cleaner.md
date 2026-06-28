# EPUB Watermark Cleaner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ebook-tts clean-epub` to remove OceanofPDF watermark artifacts while preserving the rest of an EPUB.

**Architecture:** Add a focused `epub_cleaner` module that rewrites an EPUB zip entry by entry, skipping the standalone `oceanofpdf.com` member and only changing HTML-like text entries that contain known OceanofPDF snippets. Wire that module into the existing argparse CLI and keep verification in small pytest tests plus one sample-file smoke check.

**Tech Stack:** Python 3.9+, stdlib `zipfile`, stdlib `re`, argparse, pytest.

---

## File Structure

- Create `src/ebook_tts_pipeline/epub_cleaner.py`
  - Owns output path resolution, zip metadata copying, watermark-snippet removal, and the public `clean_epub()` function.
- Modify `src/ebook_tts_pipeline/cli.py`
  - Adds `clean-epub` parser args and calls `clean_epub()`.
- Create `tests/test_epub_cleaner.py`
  - Unit-tests archive rewriting, default naming, and input overwrite protection.
- Modify `tests/test_cli.py`
  - Verifies parser support and the command's return value/output.

---

### Task 1: EPUB Cleaner Module

**Files:**
- Create: `src/ebook_tts_pipeline/epub_cleaner.py`
- Test: `tests/test_epub_cleaner.py`

- [ ] **Step 1: Write failing cleaner tests**

Create `tests/test_epub_cleaner.py`:

```python
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from ebook_tts_pipeline.epub_cleaner import clean_epub, default_cleaned_epub_path


WATERMARK_BLOCK = (
    '<div style="float: none; margin: 10px 0px 10px 0px; text-align: center;">'
    '<p><a href="https://oceanofpdf.com"><i>OceanofPDF.com</i></a></p></div>'
)


def test_clean_epub_removes_standalone_member_and_trailing_xhtml_watermark(tmp_path):
    source = tmp_path / "sample.epub"
    output = tmp_path / "sample.cleaned.epub"
    unchanged_png = b"\x89PNG\r\nunchanged"
    with ZipFile(source, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        zf.writestr("oceanofpdf.com", "oceanofpdf.com\n", compress_type=ZIP_STORED)
        zf.writestr(
            "OEBPS/xhtml/Chapter01.xhtml",
            f"<html><body><section><p>Real text.</p></section>{WATERMARK_BLOCK}</body></html>",
            compress_type=ZIP_DEFLATED,
        )
        zf.writestr("OEBPS/images/cover.png", unchanged_png, compress_type=ZIP_STORED)

    result = clean_epub(source, output)

    assert result.output_path == output
    assert result.removed_members == ["oceanofpdf.com"]
    assert result.cleaned_members == ["OEBPS/xhtml/Chapter01.xhtml"]
    with ZipFile(output) as cleaned:
        assert "oceanofpdf.com" not in cleaned.namelist()
        assert cleaned.read("OEBPS/images/cover.png") == unchanged_png
        chapter = cleaned.read("OEBPS/xhtml/Chapter01.xhtml").decode("utf-8")
        assert "Real text." in chapter
        assert "OceanofPDF.com" not in chapter
        assert "oceanofpdf.com" not in chapter.lower()
        assert cleaned.getinfo("mimetype").compress_type == ZIP_STORED


def test_clean_epub_leaves_non_watermarked_html_byte_for_byte(tmp_path):
    source = tmp_path / "sample.epub"
    output = tmp_path / "sample.cleaned.epub"
    html = b"<html><body><p>Nothing to remove.</p></body></html>"
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        zf.writestr("OEBPS/xhtml/Chapter01.xhtml", html, compress_type=ZIP_DEFLATED)

    clean_epub(source, output)

    with ZipFile(output) as cleaned:
        assert cleaned.read("OEBPS/xhtml/Chapter01.xhtml") == html


def test_default_cleaned_epub_path_adds_cleaned_suffix(tmp_path):
    assert default_cleaned_epub_path(tmp_path / "Book.epub") == tmp_path / "Book.cleaned.epub"


def test_clean_epub_refuses_to_overwrite_input(tmp_path):
    source = tmp_path / "sample.epub"
    with ZipFile(source, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")

    with pytest.raises(ValueError, match="Output path must be different"):
        clean_epub(source, source)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_epub_cleaner.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'ebook_tts_pipeline.epub_cleaner'`.

- [ ] **Step 3: Implement the cleaner module**

Create `src/ebook_tts_pipeline/epub_cleaner.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo


STANDALONE_WATERMARK_MEMBER = "oceanofpdf.com"
HTML_LIKE_SUFFIXES = {".html", ".htm", ".xhtml", ".xml"}

WATERMARK_BLOCK_RE = re.compile(
    r"\s*<div\b[^>]*>\s*<p>\s*<a\b[^>]*href=[\"']https?://(?:www\.)?oceanofpdf\.com/?[\"'][^>]*>"
    r"\s*(?:<i>)?\s*OceanofPDF\.com\s*(?:</i>)?\s*</a>\s*</p>\s*</div>",
    re.IGNORECASE,
)
WATERMARK_PARAGRAPH_RE = re.compile(
    r"\s*<p>\s*<a\b[^>]*href=[\"']https?://(?:www\.)?oceanofpdf\.com/?[\"'][^>]*>"
    r"\s*(?:<i>)?\s*OceanofPDF\.com\s*(?:</i>)?\s*</a>\s*</p>",
    re.IGNORECASE,
)
WATERMARK_LINK_RE = re.compile(
    r"<a\b[^>]*href=[\"']https?://(?:www\.)?oceanofpdf\.com/?[\"'][^>]*>"
    r"\s*(?:<i>)?\s*OceanofPDF\.com\s*(?:</i>)?\s*</a>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EpubCleanResult:
    input_path: Path
    output_path: Path
    removed_members: List[str]
    cleaned_members: List[str]


def default_cleaned_epub_path(input_path: str | Path) -> Path:
    epub_path = Path(input_path)
    return epub_path.with_name(f"{epub_path.stem}.cleaned{epub_path.suffix or '.epub'}")


def clean_epub(input_path: str | Path, output_path: str | Path | None = None) -> EpubCleanResult:
    source = Path(input_path)
    target = Path(output_path) if output_path is not None else default_cleaned_epub_path(source)
    if source.resolve() == target.resolve():
        raise ValueError("Output path must be different from input path")
    if source.suffix.lower() != ".epub":
        raise ValueError(f"Input path must be an .epub file: {source}")
    if not source.exists():
        raise FileNotFoundError(source)

    removed_members: List[str] = []
    cleaned_members: List[str] = []
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(source, "r") as input_zip, ZipFile(target, "w") as output_zip:
            for info in input_zip.infolist():
                if _is_standalone_watermark_member(info.filename):
                    removed_members.append(info.filename)
                    continue
                original = input_zip.read(info.filename)
                cleaned = _clean_member_bytes(info.filename, original)
                if cleaned != original:
                    cleaned_members.append(info.filename)
                output_zip.writestr(_copy_zip_info(info), cleaned, compress_type=_compress_type_for(info))
    except BadZipFile as exc:
        raise ValueError(f"Input path is not a readable EPUB zip: {source}") from exc

    return EpubCleanResult(
        input_path=source,
        output_path=target,
        removed_members=removed_members,
        cleaned_members=cleaned_members,
    )


def _copy_zip_info(info: ZipInfo) -> ZipInfo:
    copied = ZipInfo(filename=info.filename, date_time=info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.compress_type = _compress_type_for(info)
    return copied


def _compress_type_for(info: ZipInfo) -> int:
    if info.filename == "mimetype":
        return ZIP_STORED
    return info.compress_type


def _is_standalone_watermark_member(filename: str) -> bool:
    return filename.strip("/").lower() == STANDALONE_WATERMARK_MEMBER


def _clean_member_bytes(filename: str, content: bytes) -> bytes:
    if not _is_html_like_member(filename) or b"OceanofPDF" not in content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    cleaned = _clean_watermark_text(text)
    if cleaned == text:
        return content
    return cleaned.encode("utf-8")


def _is_html_like_member(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in HTML_LIKE_SUFFIXES


def _clean_watermark_text(text: str) -> str:
    cleaned = WATERMARK_BLOCK_RE.sub("", text)
    cleaned = WATERMARK_PARAGRAPH_RE.sub("", cleaned)
    cleaned = WATERMARK_LINK_RE.sub("", cleaned)
    return cleaned
```

- [ ] **Step 4: Run cleaner tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_epub_cleaner.py -q
```

Expected: all tests in `tests/test_epub_cleaner.py` pass.

- [ ] **Step 5: Commit module and tests**

Run:

```powershell
git add src\ebook_tts_pipeline\epub_cleaner.py tests\test_epub_cleaner.py
git commit -m "Add EPUB watermark cleaner"
```

---

### Task 2: CLI Command

**Files:**
- Modify: `src/ebook_tts_pipeline/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add this import to `tests/test_cli.py`:

```python
from zipfile import ZIP_STORED, ZipFile
```

Add these tests near the existing parser tests:

```python
def test_cli_has_clean_epub_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "clean-epub",
            "downloads/book.epub",
            "--output",
            "downloads/book.cleaned.epub",
        ]
    )

    assert args.command == "clean-epub"
    assert args.input == "downloads/book.epub"
    assert args.output == "downloads/book.cleaned.epub"
```

Add this test near the other `main()` integration tests:

```python
def test_cli_clean_epub_writes_cleaned_file(tmp_path, capsys):
    source = tmp_path / "book.epub"
    output = tmp_path / "book.cleaned.epub"
    with ZipFile(source, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        zf.writestr("oceanofpdf.com", "oceanofpdf.com\n")
        zf.writestr(
            "chapter.xhtml",
            '<html><body><p>Text.</p><p><a href="https://oceanofpdf.com"><i>OceanofPDF.com</i></a></p></body></html>',
        )

    result = main(["clean-epub", str(source), "--output", str(output)])

    assert result == 0
    captured = capsys.readouterr()
    assert "Cleaned EPUB written to" in captured.out
    with ZipFile(output) as cleaned:
        assert "oceanofpdf.com" not in cleaned.namelist()
        assert "OceanofPDF.com" not in cleaned.read("chapter.xhtml").decode("utf-8")
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_cli_has_clean_epub_command tests\test_cli.py::test_cli_clean_epub_writes_cleaned_file -q
```

Expected: fail because `clean-epub` is not a known subcommand.

- [ ] **Step 3: Wire the CLI parser and command**

In `src/ebook_tts_pipeline/cli.py`, add this import:

```python
from ebook_tts_pipeline.epub_cleaner import clean_epub
```

In `build_parser()`, before `return parser`, add:

```python
    clean_epub_parser = subparsers.add_parser("clean-epub")
    clean_epub_parser.add_argument("input")
    clean_epub_parser.add_argument("--output")
```

In `main()`, before the final unsupported-command error, add:

```python
    if args.command == "clean-epub":
        result = clean_epub(args.input, args.output)
        print(
            "Cleaned EPUB written to "
            f"{result.output_path} "
            f"({len(result.removed_members)} archive entries removed, "
            f"{len(result.cleaned_members)} content entries cleaned)"
        )
        return 0
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py::test_cli_has_clean_epub_command tests\test_cli.py::test_cli_clean_epub_writes_cleaned_file -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit CLI wiring**

Run:

```powershell
git add src\ebook_tts_pipeline\cli.py tests\test_cli.py
git commit -m "Add clean-epub CLI command"
```

---

### Task 3: Verification With Sample EPUB

**Files:**
- No code changes unless verification exposes a defect.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_epub_cleaner.py tests\test_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Clean the provided sample EPUB**

Run:

```powershell
.\.venv\Scripts\python.exe -m ebook_tts_pipeline.cli clean-epub "C:\Users\zhuyl\Downloads\Victorian_Psycho - Virginia_Feito.epub"
```

Expected: command exits `0` and writes `C:\Users\zhuyl\Downloads\Victorian_Psycho - Virginia_Feito.cleaned.epub`.

- [ ] **Step 4: Confirm no OceanofPDF markers remain**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import zipfile; p=r'C:\Users\zhuyl\Downloads\Victorian_Psycho - Virginia_Feito.cleaned.epub'; z=zipfile.ZipFile(p); hits=[n for n in z.namelist() if b'oceanofpdf' in z.read(n).lower()]; print(hits)"
```

Expected: prints `[]`.

- [ ] **Step 5: Commit any verification fixes**

If a defect was fixed during sample verification, run:

```powershell
git add src\ebook_tts_pipeline\epub_cleaner.py src\ebook_tts_pipeline\cli.py tests\test_epub_cleaner.py tests\test_cli.py
git commit -m "Fix EPUB watermark cleaner sample handling"
```

If no defect was fixed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Standalone `oceanofpdf.com` member removal: Task 1 test and implementation.
- HTML/XHTML watermark block/link cleanup: Task 1 test and implementation.
- Preserve unrelated members byte-for-byte: Task 1 test.
- Default `<stem>.cleaned.epub` output: Task 1 test and Task 3 sample command.
- Refuse input overwrite: Task 1 test and implementation.
- Python CLI command: Task 2 parser and `main()` integration tests.

Completeness scan: every task has exact files, commands, expected results, and concrete code snippets.

Type consistency: `clean_epub`, `default_cleaned_epub_path`, and `EpubCleanResult` are introduced in Task 1 before Task 2 imports and uses them.
