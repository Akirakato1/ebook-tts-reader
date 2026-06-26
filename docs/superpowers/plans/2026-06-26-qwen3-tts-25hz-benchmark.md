# Qwen3-TTS 25Hz Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a Qwen3-TTS 25Hz audiobook profile can fit on the 16 GB test GPU and measure generation speed against the existing 12Hz vLLM-Omni benchmark.

**Architecture:** Use the existing `scripts/benchmark_vllm_omni_qwen3_tts.py` sweep harness with a 25Hz model path and a copied vLLM-Omni stage profile. The experiment is blocked until public or locally available 25Hz Qwen3-TTS model/tokenizer assets exist.

**Tech Stack:** WSL Ubuntu 24.04, `/opt/ebook-vllm-omni-venv/bin/python`, vLLM-Omni, FlashAttention2, Qwen3-TTS, existing False Witness chapter 15 read-along units.

---

### Task 1: Confirm 25Hz Asset Availability

**Files:**
- Read: `models/qwen-tts/`
- Read: `models/qwen-tts/Qwen3-TTS-12Hz-1.7B-Base/config.json`
- Read: `models/qwen-tts/Qwen3-TTS-Tokenizer-12Hz/config.json`

- [ ] **Step 1: Query public Hugging Face model index**

Run:

```powershell
python -c "import json, urllib.request; url='https://huggingface.co/api/models?author=Qwen&search=Qwen3-TTS&limit=100'; print(urllib.request.urlopen(url, timeout=20).read().decode('utf-8')[:12000])"
```

Expected for runnable public 25Hz experiment: output includes repos such as `Qwen/Qwen3-TTS-25Hz-1.7B-Base` and `Qwen/Qwen3-TTS-Tokenizer-25Hz`.

- [ ] **Step 2: Query likely 25Hz repo IDs directly**

Run:

```powershell
python -c "import json, urllib.request, urllib.error; ids=['Qwen/Qwen3-TTS-25Hz-1.7B-Base','Qwen/Qwen3-TTS-25Hz-0.6B-Base','Qwen/Qwen3-TTS-Tokenizer-25Hz','Qwen/Qwen-TTS-Tokenizer-25Hz'];\
for repo in ids:\
    url='https://huggingface.co/api/models/'+repo;\
    try:\
        r=urllib.request.urlopen(url, timeout=20);\
        data=json.loads(r.read().decode('utf-8'));\
        print(repo, 'OK', data.get('id'));\
    except urllib.error.HTTPError as exc:\
        print(repo, 'HTTP', exc.code)"
```

Expected for runnable public 25Hz experiment: `OK` for at least the base model and tokenizer repos.

- [ ] **Step 3: Confirm local model family**

Run:

```powershell
Get-ChildItem -Directory models\qwen-tts -Force | Select-Object Name
```

Expected for runnable local 25Hz experiment: a local folder such as `Qwen3-TTS-25Hz-1.7B-Base` plus any matching tokenizer folder required by that model.

### Task 2: Run 25Hz Smoke Init When Assets Exist

**Files:**
- Read: `scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml`
- Read: `scripts/benchmark_vllm_omni_qwen3_tts.py`

- [ ] **Step 1: Run one-window smoke sweep**

Run from WSL when a local 25Hz model path exists:

```bash
cd "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader"
/opt/ebook-vllm-omni-venv/bin/python scripts/benchmark_vllm_omni_qwen3_tts.py \
  --book-root "test false witness real tts chapter 15" \
  --chapter chapter_015 \
  --start-chars 100 \
  --step-chars 100 \
  --max-targets 1 \
  --repeat-count 1 \
  --warmup-text Test \
  --max-vram-gb 16 \
  --model "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/models/qwen-tts/Qwen3-TTS-25Hz-1.7B-Base" \
  --stage-configs-path "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml" \
  --output-stem "vllm_omni_window_sweep_chapter_015_25hz_smoke"
```

Expected: either one benchmark row is written under `test false witness real tts chapter 15/logs/`, or the init fails with a clear vLLM-Omni/model compatibility error.

### Task 3: Run Full 25Hz Sweep When Smoke Passes

**Files:**
- Write: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_25hz_*.json`
- Write: `test false witness real tts chapter 15/logs/vllm_omni_window_sweep_chapter_015_25hz_*.csv`
- Write: `docs/benchmarks/2026-06-26-qwen3-tts-25hz-window-sweep.md`

- [ ] **Step 1: Run the 25Hz sweep**

Run:

```bash
cd "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader"
/opt/ebook-vllm-omni-venv/bin/python scripts/benchmark_vllm_omni_qwen3_tts.py \
  --book-root "test false witness real tts chapter 15" \
  --chapter chapter_015 \
  --start-chars 100 \
  --step-chars 100 \
  --repeat-count 3 \
  --warmup-text Test \
  --playback-speed 1.0 \
  --max-vram-gb 16 \
  --model "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/models/qwen-tts/Qwen3-TTS-25Hz-1.7B-Base" \
  --stage-configs-path "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml" \
  --output-stem "vllm_omni_window_sweep_chapter_015_25hz_full"
```

Expected: the sweep stops at `vram_limit`, model error, or chapter exhaustion and writes JSON/CSV results.

- [ ] **Step 2: Summarize system requirement**

Write a benchmark note that records:

```text
- Model and tokenizer family
- Stage profile
- Init VRAM
- Peak VRAM
- Generation time per target window
- Playback seconds per target window
- RTF at 1.0x
- Max smooth playback speed
- Whether it fits within 16 GB
- Whether it is suitable for audiobook mode only or also read-along
```

Expected: the summary explicitly compares the 25Hz result to the selected 12Hz result: peak VRAM about 11.9 GB, 764-char generation about 6.946 s, playback about 45.973 s, RTF about 0.151.

---

## Current Execution Result

As of 2026-06-26, public discovery did not find accessible 25Hz Qwen3-TTS model/tokenizer assets:

- `Qwen/Qwen3-TTS` Hugging Face author search listed only 12Hz Qwen3-TTS repos.
- `Qwen/Qwen3-TTS-25Hz-1.7B-Base`, `Qwen/Qwen3-TTS-25Hz-0.6B-Base`, `Qwen/Qwen3-TTS-Tokenizer-25Hz`, and `Qwen/Qwen-TTS-Tokenizer-25Hz` returned HTTP 401 to unauthenticated API requests.
- Local `models/qwen-tts/` contains only 12Hz assets.
- Local 12Hz configs specify `tokenizer_type: qwen3_tts_tokenizer_12hz` and `_frame_rate: 12.5`.
- Installed `vllm_omni` package search found no obvious `25Hz` or `qwen3_tts_tokenizer_25` string references.

Therefore, the benchmark is prepared but blocked until 25Hz model/tokenizer assets are available locally or through an authenticated source.
