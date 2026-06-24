# vLLM-Omni Qwen3-TTS Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the official vLLM-Omni Qwen3-TTS backend is materially faster than the current `qwen_tts` WSL worker for Chapter 15 read-along segments.

**Architecture:** Keep the existing `qwen_tts` backend untouched. Install vLLM-Omni into an isolated WSL venv, clone the official repo under `/opt`, run the official Qwen3-TTS offline example first, then add a local benchmark harness only after the official sample works.

**Tech Stack:** WSL Ubuntu 24.04, Python 3.12, vLLM 0.23.0, vLLM-Omni, Qwen3-TTS-12Hz-1.7B, existing Chapter 15 registry/read-along artifacts.

---

### Task 1: Isolated WSL Runtime

**Files:**
- Create externally: `/opt/ebook-vllm-omni-venv`
- Create externally: `/opt/vllm-omni`
- No repo code changes.

- [ ] **Step 1: Verify prerequisites**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- python3.12 --version
wsl.exe -d Ubuntu-24.04 -u root -- git --version
wsl.exe -d Ubuntu-24.04 -u root -- nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Expected: Python 3.12, git, and the NVIDIA GPU are visible.

- [ ] **Step 2: Create the venv**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- python3.12 -m venv /opt/ebook-vllm-omni-venv
```

Expected: exit code 0.

- [ ] **Step 3: Install uv into the isolated venv**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/python -m pip install --upgrade pip uv
```

Expected: exit code 0 and `/opt/ebook-vllm-omni-venv/bin/uv --version` works.

### Task 2: vLLM-Omni Install

**Files:**
- Create externally: `/opt/vllm-omni`
- No repo code changes.

- [ ] **Step 1: Install official vLLM version**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/uv pip install vllm==0.23.0 --torch-backend=auto
```

Expected: exit code 0.

- [ ] **Step 2: Clone vLLM-Omni**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- git clone https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni
```

Expected: exit code 0. If the directory already exists, inspect it instead of deleting it.

- [ ] **Step 3: Install vLLM-Omni editable**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/uv pip install -e /opt/vllm-omni
```

Expected: exit code 0.

- [ ] **Step 4: Import verification**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/python -c "import vllm, vllm_omni; print(vllm.__version__)"
```

Expected: prints `0.23.0` or a compatible version without import warnings.

### Task 3: Official Qwen3-TTS Smoke Test

**Files:**
- Read: `/opt/vllm-omni/examples/offline_inference/qwen3_tts/end2end.py`
- Output externally: `/opt/vllm-omni/examples/offline_inference/qwen3_tts/output_audio`

- [ ] **Step 1: Inspect the official example model path defaults**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- sed -n '1,260p' /opt/vllm-omni/examples/offline_inference/qwen3_tts/end2end.py
```

Expected: identify how model names, stage overrides, and prompt fields are configured.

- [ ] **Step 2: Run a single Base voice-clone sample**

Run from `/opt/vllm-omni/examples/offline_inference/qwen3_tts`:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/python end2end.py --query-type Base --mode-tag icl --output-dir /tmp/qwen3_tts_vllm_omni_smoke
```

Expected: a WAV file is written and the command exits 0.

### Task 4: Chapter 15 Comparable Benchmark

**Files:**
- Read: `test false witness real tts chapter 15/registry.json`
- Read: `test false witness real tts chapter 15/read_along/chapter_015.units.json`
- Create: `scripts/benchmark_vllm_omni_qwen3_tts.py`
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015.json`
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015.csv`

- [ ] **Step 1: Write a benchmark harness only after the official sample works**

Create a script that loads Chapter 15 read-along units, builds the same target windows as `benchmark-window-sweep`, sends text to the vLLM-Omni Qwen3-TTS Base path, records generation seconds, audio seconds, RTF, roles, char counts, and VRAM via `nvidia-smi`.

- [ ] **Step 2: Run a 100-400 char smoke sweep**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- /opt/ebook-vllm-omni-venv/bin/python /mnt/c/Users/zhuyl/OneDrive/Documents/Ebook\ Reader/scripts/benchmark_vllm_omni_qwen3_tts.py --book-root /mnt/c/Users/zhuyl/OneDrive/Documents/Ebook\ Reader/test\ false\ witness\ real\ tts\ chapter\ 15 --chapter chapter_015 --max-targets 4
```

Expected: JSON and CSV outputs with no failed rows.

- [ ] **Step 3: Run the full sweep to 10 GB VRAM**

Run the same script without `--max-targets`.

Expected: comparable rows to the current WSL worker sweep.

### Task 5: Decision

**Files:**
- Read: current `window_sweep_chapter_015.json`
- Read: new `vllm_omni_window_sweep_chapter_015.json`
- Modify only if benchmark proves worthwhile: backend adapter files.

- [ ] **Step 1: Compare RTF and VRAM**

Calculate speedup per target window:
```text
speedup = current_qwen_tts_rtf / vllm_omni_rtf
```

Expected: decide whether vLLM-Omni is worth integrating.

- [ ] **Step 2: Do not integrate unless smoke benchmark is clearly better**

If vLLM-Omni does not beat the current path or cannot run on this machine, document the blocker and keep the current backend.
