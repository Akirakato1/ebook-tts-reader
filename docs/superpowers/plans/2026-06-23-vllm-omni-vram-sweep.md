# vLLM-Omni VRAM Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a lower-VRAM vLLM-Omni Qwen3-TTS configuration that keeps Chapter 15 read-along generation faster than playback.

**Architecture:** Keep the existing benchmark path intact, add output isolation and peak VRAM sampling, then run comparable config variants against the same Chapter 15 windows. Variants only change vLLM-Omni deploy YAML values, so the TTS prompt, voices, text units, and timing math stay constant.

**Tech Stack:** Python 3.12 in WSL, vLLM-Omni, Qwen3-TTS-12Hz-1.7B-Base, YAML deploy configs, existing Chapter 15 read-along units and `.qvp` voice prompts.

---

### Task 1: Benchmark Output Isolation And Peak VRAM

**Files:**
- Modify: `scripts/benchmark_vllm_omni_qwen3_tts.py`

- [ ] **Step 1: Add CLI options**

Add arguments:
```python
parser.add_argument("--model", default=MODEL_NAME)
parser.add_argument("--output-stem")
parser.add_argument("--vram-poll-seconds", type=float, default=0.25)
```

Expected behavior: the default output path remains `vllm_omni_window_sweep_<chapter>.json`, and custom runs can write unique files such as `vllm_omni_window_sweep_chapter_015_len8192_seq2.json`.

- [ ] **Step 2: Use the selected model everywhere**

Replace hard-coded `MODEL_NAME` usage in `Omni(...)` and `_estimate_prompt_len(...)` with the parsed model name. Pass the model name into helper functions instead of relying on only the module constant.

Expected behavior: default runs still use `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, and future 25Hz smoke tests can pass `--model`.

- [ ] **Step 3: Add peak VRAM sampling**

Create a lightweight `VramSampler` that polls:
```python
["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
```
on a background thread during the full run.

Expected JSON fields:
```json
"device_vram_after_init_gb": 15.0,
"device_vram_peak_gb": 15.4
```

- [ ] **Step 4: Verify syntax**

Run:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "source /opt/ebook-vllm-omni-venv/bin/activate && cd /mnt/c/Users/zhuyl/OneDrive/Documents/Ebook\ Reader && python -m py_compile scripts/benchmark_vllm_omni_qwen3_tts.py"
```

Expected: exit code 0.

### Task 2: Create VRAM Variant Configs

**Files:**
- Create: `scripts/vllm_omni_qwen3_tts_16gb_len32768.yaml`
- Create: `scripts/vllm_omni_qwen3_tts_16gb_len16384.yaml`
- Create: `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq4.yaml`
- Create: `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2.yaml`
- Create: `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq1.yaml`

- [ ] **Step 1: Copy the known-good config**

Start each file from `scripts/vllm_omni_qwen3_tts_16gb.yaml`.

- [ ] **Step 2: Reduce Stage 1 capacity first**

Set Stage 1 `max_model_len`, `max_num_batched_tokens`, and `default_sampling_params.max_tokens` to the variant length: `32768`, `16384`, or `8192`.

Expected behavior: if VRAM is dominated by Stage 1 KV allocation, these variants should lower resident VRAM.

- [ ] **Step 3: Reduce concurrency after length**

For `seq2`, set both stage `max_num_seqs: 2` and CUDA graph capture sizes to `[1, 2]`.

For `seq1`, set both stage `max_num_seqs: 1` and CUDA graph capture sizes to `[1]`.

Expected behavior: lower concurrency should lower VRAM but may hurt read-along speed.

### Task 3: Run Comparable Quick Sweep

**Files:**
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_<variant>.json`
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_<variant>.csv`

- [ ] **Step 1: Run each variant with one repeat**

Use:
```powershell
wsl.exe -d Ubuntu-24.04 -u root -- bash -lc "source /opt/ebook-vllm-omni-venv/bin/activate && cd /mnt/c/Users/zhuyl/OneDrive/Documents/Ebook\ Reader && python scripts/benchmark_vllm_omni_qwen3_tts.py --book-root test\ false\ witness\ real\ tts\ chapter\ 15 --chapter chapter_015 --start-chars 100 --step-chars 100 --max-targets 7 --repeat-count 1 --warmup-text Test --playback-speed 1.0 --max-vram-gb 16 --stage-configs-path <config> --output-stem vllm_omni_window_sweep_chapter_015_<variant>"
```

Expected: each successful variant writes 7 rows, unless it fails to initialize or generate.

- [ ] **Step 2: Stop if two consecutive lower configs fail**

If two lower-capacity variants fail with OOM, missing KV capacity, or stage timeout, stop lowering capacity and summarize the failure boundary.

### Task 4: Confirm Best Candidate

**Files:**
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_<best>_confirm.json`
- Output: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_<best>_confirm.csv`

- [ ] **Step 1: Pick the candidate**

Choose the lowest-VRAM variant whose 764-char mixed-voice row remains faster than playback and has positive unit readiness margins.

- [ ] **Step 2: Run a 3-repeat confirmation**

Use the same benchmark command with:
```text
--repeat-count 3 --output-stem vllm_omni_window_sweep_chapter_015_<best>_confirm
```

Expected: stable RTF and no negative readiness margins.

### Task 5: Summarize Audiobook Implications

**Files:**
- Read: `test false witness real tts chapter 15/logs/*.json`

- [ ] **Step 1: Compare result rows**

Report for each variant:
```text
peak_vram_gb, init_vram_gb, 764-char generation_seconds, playback_seconds, rtf_at_1x, max_smooth_speed
```

- [ ] **Step 2: Interpret for read-along and audiobook generation**

State whether the vLLM-Omni stack should become:
```text
read-along high-performance backend
audiobook batch backend
25Hz candidate backend
```

Expected: a clear recommendation plus the next benchmark needed for 25Hz.
