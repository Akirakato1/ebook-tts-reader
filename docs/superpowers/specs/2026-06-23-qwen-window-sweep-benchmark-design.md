# Qwen Window Sweep Benchmark Design

## Goal

Build a repeatable benchmark that finds the critical Qwen3-TTS generation window size for read-along use. The benchmark should identify when larger windows become faster than playback, stop before GPU memory use exceeds 10 GB, and record the hardware/software profile needed to interpret system requirements for the app.

## Context

The current read-along benchmark generates one or two sentence/quote units per call. That shape is easy to debug, but it is slower than playback because fixed generation overhead dominates small calls. The older audiobook pipeline produced roughly 30 minutes of audio in roughly 30 minutes by generating larger audiobook-style windows, usually around 800-1100 characters and about 50 seconds of audio per section.

The new experiment should test whether the WSL + FlashAttention stack reaches better realtime factor as window size grows. It must benchmark at playback speed `1.0x` so results are comparable and can later be converted into user playback-speed margins.

## Experiment Shape

The sweep uses character budgets as the independent variable:

- Start at `100` target characters.
- Increase by `100` characters per run.
- Build each test window from consecutive read-along units, preserving unit boundaries.
- Stop when peak reserved or allocated VRAM reaches or exceeds `10 GB`.
- Also stop if the chapter runs out of units.
- Run one warmup generation first and exclude it from the result table.

Characters are the primary control because the existing audiobook splitter and Qwen perf logs are char-based. The benchmark also records word count and unit count for secondary analysis.

## Benchmark Inputs

The CLI command should be:

```powershell
ebook-tts benchmark-window-sweep --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-chars 100 --step-chars 100 --max-vram-gb 10 --playback-speed 1.0
```

Default values:

- `--start-chars`: `100`
- `--step-chars`: `100`
- `--max-vram-gb`: `10`
- `--playback-speed`: `1.0`
- `--generation-mode`: `balanced`
- `--warmup-chars`: `100`

The command should use the selected configured TTS backend. For the target experiment this means:

- `EBOOK_TTS_BACKEND=wsl`
- `EBOOK_TTS_QWEN_ATTENTION=auto`
- `EBOOK_TTS_QWEN_PRECISION=bf16`
- `EBOOK_TTS_QWEN_ADAPTIVE_TARGET_GB=10`

## Window Construction

For each target character size:

1. Start at the first chapter unit after the warmup units.
2. Add consecutive units until adding the next unit would exceed the target.
3. If the first unit alone exceeds the target, use that single unit.
4. Preserve roles, role IDs, voice paths, and source offsets.
5. Generate that window in one adapter call.

This design intentionally tests throughput windows, not UI buffer windows. The read-along UI can later use the winning window size internally while still highlighting sentence-by-sentence.

## Metrics

Each sweep row must record:

- `target_chars`
- `actual_chars`
- `word_count`
- `unit_count`
- `unit_ids`
- `role_ids`
- `voice_config_paths`
- `generation_seconds`
- `playback_seconds`
- `rtf_at_1x`, computed as `generation_seconds / playback_seconds`
- `max_smooth_speed`, computed as `playback_seconds / generation_seconds`
- `peak_vram_reserved_gb`
- `peak_vram_allocated_gb`
- `text_char_max`
- `audio_seconds`
- `success`
- `error`, when present

The benchmark should also copy or reference the Qwen performance-log row for each sweep point, because that log contains lower-level CUDA memory and block metadata.

## Machine Profile

Each benchmark output must include a machine profile so the results can later become a system-requirements table. Record:

- OS mode: Windows host + WSL distro
- GPU name
- GPU total VRAM
- CUDA runtime version reported by PyTorch
- PyTorch version
- FlashAttention version
- Qwen model variant
- Precision
- Attention mode resolved by Qwen runtime
- Python executable used by the worker
- CPU model if available
- System RAM if available

The first version only needs to record this profile for the local machine. Later machines can run the same command and append comparable profiles.

## Outputs

The command should write:

- `logs/window_sweep_chapter_015.csv`
- `logs/window_sweep_chapter_015.json`
- `docs/benchmarks/2026-06-23-chapter-15-window-sweep.md`

The markdown summary should include:

- The command and environment variables used.
- The machine profile.
- A table of sweep results.
- The first window size where `rtf_at_1x <= 1.0`.
- The first window size where `rtf_at_1x <= 0.85`.
- The first window size where `rtf_at_1x <= 0.7`.
- The largest window completed before reaching 10 GB VRAM.
- A conclusion about whether local Qwen can support seamless live read-along at `1.0x`, `1.25x`, and `1.5x`.

## Success Criteria

The implementation is successful when:

- The sweep runs on chapter 15 using the WSL backend.
- Results include at least one row per 100-character increment until stop.
- The stop reason is explicit: `vram_limit`, `chapter_exhausted`, or `generation_error`.
- The benchmark excludes warmup from the result table.
- Every row includes generation time, playback time, RTF, max smooth speed, and VRAM.
- The markdown summary identifies the critical threshold windows for `1.0`, `0.85`, and `0.7` RTF targets.

## Non-Goals

- Do not change the read-along UI behavior in this benchmark task.
- Do not replace the TTS backend with vLLM-Omni in this task.
- Do not infer universal minimum hardware requirements from one machine. The output should describe the local measured system and make it easy to compare future machines.

## Open Implementation Notes

The implementation should reuse `ReadAlongUnit` and the existing TTS adapter interfaces. It should not duplicate the audiobook synthesis path. The benchmark should generate windows directly through the adapter and calculate playback seconds from returned sample counts at playback speed `1.0x`.
