# Chapter 15 WSL FlashAttention Benchmark

Command, balanced run:
`ebook-tts benchmark-readalong --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-unit 0 --unit-count 20 --target-buffer-seconds 12 --generation-mode balanced`

Command, precise timing sample:
`ebook-tts benchmark-readalong --book-root "test false witness real tts chapter 15" --chapter chapter_015 --start-unit 0 --unit-count 8 --target-buffer-seconds 12 --generation-mode precise`

Environment:
- Backend: WSL Ubuntu-24.04
- Torch: 2.9.1+cu130
- FlashAttention: 2.8.3
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- Precision: bf16
- Attention: flash_attention_2
- Adaptive memory target: 13 GB

Balanced 20-unit results:
- Timing log: `test false witness real tts chapter 15\read_along_sessions\benchmark-chapter_015-1782198592\timings.jsonl`
- Calls: 13
- Generation seconds: 202.624
- Playback seconds: 86.659
- Realtime factor: 2.338
- First call seconds: 47.462
- Steady generation seconds, excluding first call: 155.162
- Steady playback seconds, excluding first call: 77.963
- Steady realtime factor: 1.990
- Max reserved VRAM observed in perf log: about 5.23 GB

Precise 8-unit per-sentence sample:
- Timing log: `test false witness real tts chapter 15\read_along_sessions\benchmark-chapter_015-1782198853\timings.jsonl`
- Calls: 8
- Generation seconds: 113.001
- Playback seconds: 43.895
- Realtime factor: 2.574
- First call seconds: 44.808
- Steady generation seconds, excluding first call: 68.193
- Steady playback seconds, excluding first call: 35.838
- Steady realtime factor: 1.903

Precise per-unit timing:

| Unit | Role | Chars | Generation s | Playback s | Realtime Factor | Voice |
|---:|---|---:|---:|---:|---:|---|
| 0 | narrator | 122 | 44.808 | 8.057 | 5.562 | voices/narrator.qvp |
| 1 | narrator | 21 | 3.491 | 1.417 | 2.464 | voices/narrator.qvp |
| 2 | narrator | 21 | 2.556 | 1.257 | 2.033 | voices/narrator.qvp |
| 3 | narrator | 96 | 13.901 | 7.417 | 1.874 | voices/narrator.qvp |
| 4 | narrator | 103 | 11.608 | 6.137 | 1.892 | voices/narrator.qvp |
| 5 | narrator | 74 | 7.402 | 3.817 | 1.939 | voices/narrator.qvp |
| 6 | narrator | 223 | 18.374 | 10.537 | 1.744 | voices/narrator.qvp |
| 7 | maddy_collier_teen | 80 | 10.860 | 5.257 | 2.066 | voices/maddy_collier_teen.qvp |

Conclusion:
- Current WSL FlashAttention generation is stable and uses the expected voice paths in these runs.
- The current path is not seamless for live read-along. Even after excluding first-call model startup, generation is about 1.9-2.0x slower than playback.
- VRAM is not the bottleneck in this benchmark. Max reserved memory stayed near 5.23 GB despite a 13 GB target, so the next optimization should focus on model decode throughput, warm persistent worker lifetime across the whole read-along session, and avoiding unnecessary model/process startup.
