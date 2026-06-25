# Readalongweb Python Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the project as a pip-installable `readalongweb` application whose only public UI is the local web read-along app.

**Architecture:** Keep the existing `src/ebook_tts_pipeline` import package for now, but make the distribution and command user-facing around `readalongweb`. The wheel installs the web server, EPUB/annotation/read-along code, and lightweight Python dependencies; GPU TTS stacks, WSL environments, and model weights are verified by a doctor command and configured explicitly instead of being silently downloaded during app use.

**Tech Stack:** Python packaging with `pyproject.toml`/setuptools, console scripts, stdlib local HTTP server, Anthropic SDK, NLTK/numpy, optional native Qwen extras, external WSL vLLM-Omni/Qwen runtime, pytest, `python -m build`.

---

## File Structure

- Modify: `pyproject.toml`
  - Rename/position the installable distribution for the web app.
  - Keep `readalongweb` as the main console command.
  - Add a `readalongweb-doctor` console command.
  - Keep heavy TTS dependencies out of required dependencies.
- Modify: `README.md`
  - Replace editable-dev install instructions with user install, dev install, and GPU setup sections.
  - Document that the app is launched from a library folder with `readalongweb`.
  - Document that Tkinter is not a supported UI path.
- Create: `src/ebook_tts_pipeline/ui/doctor.py`
  - Validate Python package dependencies, configured model paths, WSL availability, WSL Python paths, stage config paths, and environment variables.
  - Print readable output by default and JSON with `--json`.
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
  - Keep `readalongweb` launch behavior stable.
  - Add a CLI self-check hook that warns about missing model/runtime configuration before the user starts expensive actions, without blocking fake-TTS UI testing.
- Modify: `src/ebook_tts_pipeline/config.py`
  - Make model/runtime paths explicit and package-safe.
  - Avoid resolving relative model paths against an editable source tree in installed mode unless the user has set that path.
- Modify: `src/ebook_tts_pipeline/tts/vllm_omni_adapter.py`
  - Ensure installed-package execution can locate packaged stage configs by `importlib.resources`, while still allowing explicit env overrides.
- Move or include: `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml`
  - Either package it under `src/ebook_tts_pipeline/assets/vllm_omni/` or keep it as external documented config and require `EBOOK_TTS_VLLM_OMNI_STAGE_CONFIGS_PATH`.
  - Recommended: package it as an asset so `readalongweb` works after wheel install.
- Create: `src/ebook_tts_pipeline/assets/vllm_omni/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml`
  - Packaged copy of the known-good balanced read-along profile.
- Modify: `tests/test_cli.py`
  - Keep assertions that `readalongweb` exists and `ebook-tts-ui` does not.
- Create: `tests/test_packaging_metadata.py`
  - Validate public scripts, dependency boundaries, and packaged assets.
- Create: `tests/test_readalongweb_doctor.py`
  - Validate doctor output for missing config, fake configured config, and JSON mode.
- Create: `tests/test_installed_mode_paths.py`
  - Validate default stage config resolution through package resources.

## Task 1: Lock Public Package Identity And Scripts

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_cli.py`
- Create: `tests/test_packaging_metadata.py`

- [ ] **Step 1: Write failing metadata tests**

```python
# tests/test_packaging_metadata.py
from pathlib import Path
import tomllib


def load_project_metadata():
    with Path("pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_distribution_is_readalongweb_first():
    project = load_project_metadata()
    assert project["name"] == "readalongweb"
    assert "local web read-along" in project["description"].lower()


def test_public_console_scripts_are_web_only():
    project = load_project_metadata()
    scripts = project["scripts"]
    assert scripts["readalongweb"] == "ebook_tts_pipeline.ui.web_app:main"
    assert scripts["readalongweb-doctor"] == "ebook_tts_pipeline.ui.doctor:main"
    assert "ebook-tts-ui" not in scripts


def test_required_dependencies_do_not_pull_gpu_tts_stack():
    project = load_project_metadata()
    required = "\n".join(project.get("dependencies", [])).lower()
    blocked = ["torch", "torchaudio", "transformers", "flash-attn", "vllm", "qwen3-tts-comfyui"]
    assert all(name not in required for name in blocked)
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_packaging_metadata.py -q
```

Expected: FAIL because the project name is currently `ebook-tts-pipeline` and `readalongweb-doctor` does not exist.

- [ ] **Step 3: Update package metadata**

Change `pyproject.toml` to:

```toml
[project]
name = "readalongweb"
version = "0.1.0"
description = "Notebook-style local web read-along app for EPUB TTS"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "anthropic>=0.55",
  "nltk>=3.9",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
  "build>=1.2",
  "pytest>=8.0",
]
qwen-native = [
  "accelerate",
  "hf_xet",
  "librosa",
  "numpy>=1.26,<2",
  "onnxruntime-openvino",
  "sox",
  "soundfile",
  "torch",
  "torchaudio",
  "transformers==4.57.3",
]

[project.scripts]
ebook-tts = "ebook_tts_pipeline.cli:main"
ebook-tts-readalong-web = "ebook_tts_pipeline.ui.web_app:main"
readalongweb = "ebook_tts_pipeline.ui.web_app:main"
readalongweb-doctor = "ebook_tts_pipeline.ui.doctor:main"
```

The `qwen3-tts-comfyui @ git+https://...` dependency must not be in package metadata for a publishable wheel. Put that WSL/native setup in the doctor/runbook path instead.

- [ ] **Step 4: Run the metadata tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_packaging_metadata.py tests/test_cli.py::test_pyproject_exposes_readalongweb_script -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml tests/test_packaging_metadata.py tests/test_cli.py
git commit -m "package readalongweb as the public app"
```

## Task 2: Add A Doctor Command For Install And Runtime Checks

**Files:**
- Create: `src/ebook_tts_pipeline/ui/doctor.py`
- Create: `tests/test_readalongweb_doctor.py`

- [ ] **Step 1: Write failing doctor tests**

```python
# tests/test_readalongweb_doctor.py
import json
from pathlib import Path

from ebook_tts_pipeline.ui import doctor


def test_doctor_reports_missing_model_root(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing-qwen"
    monkeypatch.setenv("EBOOK_TTS_QWEN_MODEL_ROOT", str(missing))

    code = doctor.main(["--no-wsl"])

    output = capsys.readouterr().out
    assert code == 1
    assert "Qwen model root" in output
    assert str(missing) in output


def test_doctor_json_reports_ok_when_minimum_paths_exist(tmp_path, monkeypatch, capsys):
    model_root = tmp_path / "models" / "qwen-tts"
    for name in [
        "Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-Tokenizer-12Hz",
    ]:
        (model_root / name).mkdir(parents=True)
    monkeypatch.setenv("EBOOK_TTS_QWEN_MODEL_ROOT", str(model_root))

    code = doctor.main(["--json", "--no-wsl"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["checks"]["qwen_model_root"]["ok"] is True
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_readalongweb_doctor.py -q
```

Expected: FAIL with `ImportError` because `ebook_tts_pipeline.ui.doctor` does not exist.

- [ ] **Step 3: Implement the doctor command**

Create `src/ebook_tts_pipeline/ui/doctor.py`:

```python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ebook_tts_pipeline.config import PipelineConfig, resolve_qwen_model_root


REQUIRED_QWEN_12HZ_DIRS = [
    "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen3-TTS-Tokenizer-12Hz",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readalongweb-doctor")
    parser.add_argument("--book-root", default=".")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-wsl", action="store_true")
    return parser


def check_model_root(config: PipelineConfig) -> Dict[str, Any]:
    root = resolve_qwen_model_root(config.qwen_model_root)
    missing = [name for name in REQUIRED_QWEN_12HZ_DIRS if not (root / name).exists()]
    return {
        "ok": root.exists() and not missing,
        "path": str(root),
        "missing": missing,
        "message": "Qwen model root is ready." if root.exists() and not missing else f"Qwen model root is incomplete: {root}",
    }


def check_wsl(config: PipelineConfig) -> Dict[str, Any]:
    if shutil.which("wsl.exe") is None:
        return {"ok": False, "message": "wsl.exe was not found on PATH."}
    result = subprocess.run(
        ["wsl.exe", "-d", config.wsl_distro, "--", "test", "-x", config.vllm_omni_wsl_python],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": result.returncode == 0,
        "distro": config.wsl_distro,
        "python": config.vllm_omni_wsl_python,
        "message": "WSL TTS Python is available." if result.returncode == 0 else "WSL TTS Python is missing or not executable.",
    }


def run_checks(book_root: str, include_wsl: bool = True) -> Dict[str, Any]:
    config = PipelineConfig.from_env(book_root=book_root)
    checks: Dict[str, Any] = {"qwen_model_root": check_model_root(config)}
    if include_wsl:
        checks["wsl_tts_runtime"] = check_wsl(config)
    ok = all(item["ok"] for item in checks.values())
    return {"ok": ok, "checks": checks}


def render_text(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    for name, check in payload["checks"].items():
        status = "OK" if check["ok"] else "FAIL"
        lines.append(f"[{status}] {name}: {check['message']}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_checks(book_root=args.book_root, include_wsl=not args.no_wsl)
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(payload), end="")
    return 0 if payload["ok"] else 1
```

- [ ] **Step 4: Run doctor tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_readalongweb_doctor.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/ebook_tts_pipeline/ui/doctor.py tests/test_readalongweb_doctor.py
git commit -m "add readalongweb install doctor"
```

## Task 3: Package The Balanced vLLM-Omni Stage Config

**Files:**
- Create: `src/ebook_tts_pipeline/assets/vllm_omni/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml`
- Modify: `pyproject.toml`
- Modify: `src/ebook_tts_pipeline/config.py`
- Modify: `src/ebook_tts_pipeline/tts/vllm_omni_adapter.py`
- Create: `tests/test_installed_mode_paths.py`

- [ ] **Step 1: Write failing path tests**

```python
# tests/test_installed_mode_paths.py
from pathlib import Path

from ebook_tts_pipeline.config import PipelineConfig, resolve_project_path


def test_default_vllm_stage_config_resolves_inside_package():
    config = PipelineConfig.from_env(book_root="books/demo", user_env_lookup=lambda name: None)
    path = resolve_project_path(config.vllm_omni_stage_configs_path)
    assert path.exists()
    assert path.name == "vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml"
    assert "assets" in path.parts
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_installed_mode_paths.py -q
```

Expected: FAIL because the default currently points at `scripts/...` in the source checkout.

- [ ] **Step 3: Copy the known-good YAML into package assets**

Create `src/ebook_tts_pipeline/assets/vllm_omni/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml` with the same contents as `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml`.

- [ ] **Step 4: Include package data**

Add this to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
ebook_tts_pipeline = [
  "assets/vllm_omni/*.yaml",
]
```

- [ ] **Step 5: Resolve the default packaged asset**

In `src/ebook_tts_pipeline/config.py`, replace the default stage config construction with an importlib-resources-safe path:

```python
from importlib import resources


def default_vllm_omni_stage_config() -> str:
    return str(
        resources.files("ebook_tts_pipeline")
        / "assets"
        / "vllm_omni"
        / "vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml"
    )


DEFAULT_VLLM_OMNI_STAGE_CONFIG = default_vllm_omni_stage_config()
```

- [ ] **Step 6: Run installed-mode path test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_installed_mode_paths.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml src/ebook_tts_pipeline/config.py src/ebook_tts_pipeline/assets/vllm_omni tests/test_installed_mode_paths.py
git commit -m "package readalong tts stage config"
```

## Task 4: Make Web Launch Notebook-Like After Wheel Install

**Files:**
- Modify: `src/ebook_tts_pipeline/ui/web_app.py`
- Modify: `tests/test_read_along_web_app.py`

- [ ] **Step 1: Write launch behavior tests**

```python
from pathlib import Path

from ebook_tts_pipeline.ui.web_app import resolve_launch_root


def test_launch_root_uses_books_child_when_present(tmp_path):
    books = tmp_path / "books"
    books.mkdir()

    selection = resolve_launch_root(tmp_path)

    assert selection.library_root == books.resolve()
    assert selection.active_book_root is None


def test_launch_root_uses_current_directory_when_no_books_child(tmp_path):
    selection = resolve_launch_root(tmp_path)

    assert selection.library_root == tmp_path.resolve()
    assert selection.active_book_root is None
```

- [ ] **Step 2: Run launch tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_read_along_web_app.py::test_launch_root_uses_books_child_when_present tests/test_read_along_web_app.py::test_launch_root_uses_current_directory_when_no_books_child -q
```

Expected: PASS if current behavior is already correct. If it fails, adjust only `resolve_launch_root()`.

- [ ] **Step 3: Add a startup diagnostic line for installed mode**

In `run_server()` after printing the URL, print the resolved library/book root and fake-TTS state:

```python
print(
    f"[ebook-tts] readalongweb_launch library_root={server.app_state.library_root} "
    f"book_root={server.app_state.book_root} fake_tts={fake_tts}",
    flush=True,
)
```

- [ ] **Step 4: Run web app tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_read_along_web_app.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/ebook_tts_pipeline/ui/web_app.py tests/test_read_along_web_app.py
git commit -m "harden notebook style readalongweb launch"
```

## Task 5: Remove Tkinter As A Public Or Documented UI

**Files:**
- Modify: `README.md`
- Modify: `tests/test_packaging_metadata.py`

- [ ] **Step 1: Add a source scan test**

Append to `tests/test_packaging_metadata.py`:

```python
def test_source_tree_has_no_tkinter_ui_imports():
    python_files = Path("src/ebook_tts_pipeline").rglob("*.py")
    offenders = [
        str(path)
        for path in python_files
        if "tkinter" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
```

- [ ] **Step 2: Run the test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_packaging_metadata.py::test_source_tree_has_no_tkinter_ui_imports -q
```

Expected: PASS in the current tree. If it fails, remove the Tkinter file/import and any script entrypoint pointing to it.

- [ ] **Step 3: Update README UI wording**

Make these README statements explicit:

```markdown
The only supported UI is `readalongweb`, the local browser app. The older Tkinter prototype is not installed, documented, or maintained. Audiobook generation will be added to the web app instead of revived as a separate desktop UI.
```

- [ ] **Step 4: Run README-adjacent metadata tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_packaging_metadata.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add README.md tests/test_packaging_metadata.py
git commit -m "document web ui as the only supported interface"
```

## Task 6: Document User Install, Dev Install, And GPU Runtime Setup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace Quick Start with package install flow**

Add:

````markdown
## User Install

```powershell
py -m pip install readalongweb
readalongweb
```

Run the command from the folder that should act as your book library. If the folder has a `books/` child, that child is used as the library root.

```powershell
cd "C:\Users\you\Documents\Ebooks"
readalongweb
```
````

- [ ] **Step 2: Add dev install flow**

Add:

````markdown
## Development Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
readalongweb books --fake-tts
```
````

- [ ] **Step 3: Add runtime doctor flow**

Add:

````markdown
## Runtime Check

```powershell
readalongweb-doctor
```

The doctor verifies the configured Qwen model folder, WSL runtime, and local TTS Python path. It does not download models during normal app launch or session start.
````

- [ ] **Step 4: Explain dependency boundary honestly**

Add:

```markdown
The pip package installs the web app and Python orchestration dependencies. It does not bundle NVIDIA drivers, CUDA, WSL, vLLM-Omni, flash-attn, or Qwen model weights. Those are machine/runtime assets and must be installed once, then pointed to with environment variables such as `EBOOK_TTS_QWEN_MODEL_ROOT` and `EBOOK_TTS_VLLM_OMNI_WSL_PYTHON`.
```

- [ ] **Step 5: Commit**

```powershell
git add README.md
git commit -m "document readalongweb package install"
```

## Task 7: Build And Install The Wheel In A Clean Environment

**Files:**
- Modify only if smoke testing finds a packaging defect.

- [ ] **Step 1: Build the wheel**

Run:

```powershell
.\.venv\Scripts\python.exe -m build
```

Expected: `dist/readalongweb-0.1.0-py3-none-any.whl` and `dist/readalongweb-0.1.0.tar.gz` are created.

- [ ] **Step 2: Install in a clean venv**

Run:

```powershell
py -m venv C:\tmp\readalongweb-wheel-smoke
C:\tmp\readalongweb-wheel-smoke\Scripts\python.exe -m pip install --upgrade pip
C:\tmp\readalongweb-wheel-smoke\Scripts\python.exe -m pip install dist\readalongweb-0.1.0-py3-none-any.whl
```

Expected: install succeeds without installing torch, transformers, vLLM, or qwen Git dependencies.

- [ ] **Step 3: Smoke test console commands**

Run:

```powershell
C:\tmp\readalongweb-wheel-smoke\Scripts\readalongweb.exe --help
C:\tmp\readalongweb-wheel-smoke\Scripts\readalongweb-doctor.exe --json --no-wsl
```

Expected:
- `readalongweb --help` prints the launch options.
- `readalongweb-doctor --json --no-wsl` exits nonzero if local models are missing, but prints valid JSON instead of crashing.

- [ ] **Step 4: Smoke test fake web launch**

Run:

```powershell
C:\tmp\readalongweb-wheel-smoke\Scripts\readalongweb.exe C:\tmp --fake-tts --no-open --port 0
```

Expected: terminal prints `Read-along web UI: http://127.0.0.1:<port>/` and `[ebook-tts] readalongweb_launch ...`. Stop with `Ctrl+C`.

- [ ] **Step 5: Run full tests in dev venv**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit packaging fixes if any were required**

```powershell
git add pyproject.toml README.md src tests
git commit -m "fix readalongweb wheel smoke issues"
```

Only run this commit step if Step 1-5 required code or docs changes.

## Task 8: Prepare PyPI/TestPyPI Release Workflow

**Files:**
- Create: `docs/runbooks/release-readalongweb.txt`

- [ ] **Step 1: Write release runbook**

Create `docs/runbooks/release-readalongweb.txt`:

```text
Readalongweb release checklist

1. Verify clean tree:
   git status --short

2. Run tests:
   .\.venv\Scripts\python.exe -m pytest -q

3. Build package:
   .\.venv\Scripts\python.exe -m build

4. Install wheel in clean venv:
   py -m venv C:\tmp\readalongweb-wheel-smoke
   C:\tmp\readalongweb-wheel-smoke\Scripts\python.exe -m pip install --upgrade pip
   C:\tmp\readalongweb-wheel-smoke\Scripts\python.exe -m pip install dist\readalongweb-0.1.0-py3-none-any.whl

5. Verify console commands:
   C:\tmp\readalongweb-wheel-smoke\Scripts\readalongweb.exe --help
   C:\tmp\readalongweb-wheel-smoke\Scripts\readalongweb-doctor.exe --json --no-wsl

6. Publish to TestPyPI first:
   .\.venv\Scripts\python.exe -m twine upload --repository testpypi dist/*

7. Install from TestPyPI in a new venv and repeat command checks.

8. Publish to PyPI:
   .\.venv\Scripts\python.exe -m twine upload dist/*
```

- [ ] **Step 2: Add release dependencies to dev extra**

Add to `pyproject.toml` dev extra:

```toml
"twine>=5.0",
```

- [ ] **Step 3: Commit release docs**

```powershell
git add pyproject.toml docs/runbooks/release-readalongweb.txt
git commit -m "add readalongweb release runbook"
```

## Self-Review

- Spec coverage:
  - Pip-installable app: Task 1, Task 7, Task 8.
  - Notebook-like `readalongweb` command: Task 1, Task 4, Task 6.
  - Only web read-along UI: Task 1, Task 5, Task 6.
  - Tkinter discarded as public UI: Task 5.
  - Audiobook generation eventually belongs in web UI: Task 5 README wording.
  - Dependencies installed like a pip project: Task 1 installs Python orchestration dependencies; Task 2 and Task 6 explicitly handle GPU/model/WSL runtime assets that pip should not silently download.
- Placeholder scan:
  - No task depends on `TBD`, `TODO`, or an unspecified implementation.
- Type/signature consistency:
  - `doctor.main(argv: Optional[Iterable[str]] = None) -> int` is used by both console script and tests.
  - `run_checks(book_root: str, include_wsl: bool = True)` is used by `main()` and can be tested directly.
  - `readalongweb` remains `ebook_tts_pipeline.ui.web_app:main`.
