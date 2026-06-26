# Qwen3-TTS 25Hz Availability Check

Date: 2026-06-26

## Goal

Set up and run a 25Hz Qwen3-TTS vLLM-Omni benchmark analogous to the selected 12Hz read-along sweep.

## Result

The 25Hz benchmark could not be conducted because no accessible 25Hz Qwen3-TTS model/tokenizer assets were found.
The smoke benchmark was still executed against the expected 25Hz 1.7B Base repo ID, and it failed during vLLM-Omni model initialization with Hugging Face `401 Unauthorized` / repository-not-found.

## Evidence

Hugging Face author search for `Qwen3-TTS` under `Qwen` returned only public 12Hz repos:

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-Tokenizer-12Hz`

Likely 25Hz repo IDs were checked directly and returned HTTP 401 to unauthenticated requests:

- `Qwen/Qwen3-TTS-25Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-25Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-Tokenizer-25Hz`
- `Qwen/Qwen-TTS-Tokenizer-25Hz`

Local model folders under `models/qwen-tts/` are also 12Hz-only:

- `Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen3-TTS-Tokenizer-12Hz`

The local 12Hz base model config contains:

```json
{
  "tokenizer_type": "qwen3_tts_tokenizer_12hz",
  "talker_config": {
    "position_id_per_seconds": 13
  }
}
```

The tokenizer config contains:

```json
{
  "model_type": "qwen3_tts_tokenizer_12hz",
  "encoder_config": {
    "_frame_rate": 12.5
  }
}
```

A search through the installed WSL `vllm_omni` Python package found no obvious `25Hz`, `25hz`, `qwen3_tts_tokenizer_25`, or `Tokenizer-25` references.

## Conclusion

The project can keep 25Hz as an audiobook-only experimental setting in the UI design, but it should not be enabled as a runnable mode until one of these is true:

1. Qwen publishes public 25Hz Qwen3-TTS base/tokenizer assets compatible with vLLM-Omni.
2. The user provides a local 25Hz model folder with matching config/tokenizer assets.
3. The user provides authenticated access to private/gated 25Hz Hugging Face assets.

Until then, the supported accelerated path remains the 12Hz vLLM-Omni profile:

```text
scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml
```

Selected 12Hz reference result:

- Peak VRAM: about 11.937 GB
- 764-character generation time: about 6.946 s
- Playback time: about 45.973 s
- RTF: about 0.151
- Smooth speed ceiling: about 6.62x

## Prepared Follow-Up

The runnable 25Hz benchmark plan is saved at:

```text
docs/superpowers/plans/2026-06-26-qwen3-tts-25hz-benchmark.md
```

The conducted smoke-attempt output is saved at:

```text
docs/benchmarks/2026-06-26-qwen3-tts-25hz-smoke-attempt.txt
```
