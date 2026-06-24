# chapter_015 Qwen Window Sweep

## Machine Profile

- cuda_available: True
- cuda_version: 13.0
- flash_attn_version: 2.8.3
- gpu_name: NVIDIA GeForce RTX 5080 Laptop GPU
- gpu_total_vram_gb: 15.92
- host_platform: Windows-10-10.0.26200-SP0
- host_processor: Intel64 Family 6 Model 198 Stepping 2, GenuineIntel
- qwen_attention: auto
- qwen_model_choice: 1.7B
- qwen_model_root: models/qwen-tts
- qwen_precision: bf16
- torch_version: 2.9.1+cu130
- tts_backend: wsl
- wsl_distro: Ubuntu-24.04
- wsl_python: /opt/ebook-tts-venv/bin/python

## Results

- Stop reason: `vram_limit`
- Repeat count per target: `3`
- First RTF <= 1.0: not reached
- First RTF <= 0.85: not reached
- First RTF <= 0.7: not reached

| Target Chars | Actual Chars | Units | Repeats | Gen s | Playback s | RTF | Max Smooth Speed | Peak VRAM Reserved GB | Device VRAM Used After GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 122 | 1 | 3 | 8.997 | 7.764 | 1.159 | 0.863 | 4.312 | 5.593 |
| 200 | 164 | 3 | 3 | 18.754 | 11.710 | 1.602 | 0.624 | 4.498 | 5.779 |
| 300 | 260 | 4 | 3 | 22.763 | 16.590 | 1.372 | 0.729 | 4.844 | 6.125 |
| 400 | 363 | 5 | 3 | 30.729 | 22.190 | 1.385 | 0.722 | 5.266 | 6.546 |
| 500 | 437 | 6 | 3 | 33.956 | 26.247 | 1.294 | 0.773 | 5.068 | 6.349 |
| 700 | 660 | 7 | 3 | 46.052 | 34.567 | 1.332 | 0.751 | 5.107 | 6.388 |
| 800 | 764 | 10 | 3 | 50.507 | 43.607 | 1.158 | 0.863 | 8.984 | 10.267 |
