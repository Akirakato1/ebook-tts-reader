# Ebook Audiobook Pipeline Design

Date: 2026-06-20

## Purpose

Build a Python-first proof pipeline for chapter-by-chapter audiobook generation. The first version validates the core TTS stack before building the ebook reader UI.

The pipeline turns book text into deterministic chapter sentence artifacts, uses Anthropic to label each sentence by role and speech type, creates persistent Qwen3-TTS voice profiles, generates chapter audio, and writes sentence timing metadata for later reader-side highlighting.

## Scope

Included in v1:

- Whole-book `.txt` ingestion with deterministic chapter splitting when confidence is high.
- Pre-split chapter `.txt` ingestion.
- Deterministic sentence segmentation saved as canonical per-chapter artifacts.
- Anthropic LLM annotation, defaulting to Haiku, with model override through configuration.
- Strict validation and one repair retry for invalid LLM JSON.
- Per-book `registry.json` with narrator, character profiles, voice profiles, voice identity differentiators, and `.qvp` paths.
- Automatic voice creation for characters whose `.qvp` file is missing.
- Direct Python wrapper around the `qwen_tts` model code from `ComfyUI-Qwen-TTS`.
- Per-sentence TTS generation, stitching, and sentence-level timing metadata.
- WAV output first, with MP3 as optional later configuration.
- Unit tests with fake Anthropic and fake Qwen adapters.
- A manual runbook for local prerequisites and operation.

Not included in v1:

- Ebook reader UI.
- Interactive review/edit UI for new character profiles.
- EPUB parsing.
- ComfyUI workflow execution.
- LM Studio.
- A custom FastAPI TTS service.
- Word-level alignment.

## Architecture

```text
source book or chapter text
  -> deterministic chapter splitter
  -> deterministic sentence segmenter
  -> sentence_segments/*.sentences.json
  -> Anthropic annotation pass
  -> registry update
  -> Qwen voice creation for missing .qvp files
  -> per-sentence Qwen TTS generation
  -> stitched chapter WAV
  -> sentence timeline JSON
```

The sentence segment artifact is the canonical downstream input. Raw chapter text remains for traceability and re-segmentation, but annotation and TTS do not read raw chapter text.

## Book Folder Layout

```text
books/<book_slug>/
  source/
    book.txt
  chapters/
    chapter_001.txt
    chapter_002.txt
  sentence_segments/
    chapter_001.sentences.json
    chapter_002.sentences.json
  annotations/
    chapter_001.annotation.json
  audio/
    chapter_001.wav
    chapter_001.timeline.json
  voices/
    narrator.qvp
    narrator.json
    elena.qvp
    elena.json
  registry.json
```

If `chapters/*.txt` already exists, v1 uses those files directly. If only `source/book.txt` exists, v1 attempts deterministic chapter splitting before sentence segmentation.

## Ingestion And Segmentation

Chapter splitting must avoid LLM use. The splitter uses configurable regex patterns for headings such as:

- `Chapter 1`
- `CHAPTER ONE`
- `1.`
- `Prologue`
- `Epilogue`
- `Part I`

The splitter only writes chapter files when confidence is high. A high-confidence split requires at least two plausible chapter headings and reasonable chapter lengths. If confidence is low, the pipeline fails with a clear message asking for pre-split chapters or a custom chapter heading regex.

Sentence segmentation is deterministic, using a library such as `nltk.sent_tokenize`. Each chapter produces a stable artifact:

```json
{
  "chapter": "chapter_001",
  "source_path": "chapters/chapter_001.txt",
  "segmenter": {
    "name": "nltk.sent_tokenize",
    "language": "english",
    "version": "recorded-at-runtime"
  },
  "sentences": [
    {"idx": 0, "text": "The room was silent."},
    {"idx": 1, "text": "\"Elena,\" Marcus said."}
  ]
}
```

All later artifacts refer back to `sentence_idx`.

## Annotation

The annotation pass calls Anthropic directly. The default model is Haiku, with the pinned model ID stored in configuration and an environment/config override for experiments. The prompt receives:

- Known registry characters and aliases.
- The current chapter ID.
- A contiguous window of saved sentences rendered as `[idx] text`.
- The required JSON schema.

The model returns token-efficient JSON:

```json
{
  "new_characters": [
    {
      "name": "Marcus",
      "profile": {
        "age_range": "elderly",
        "gender": "male",
        "personality": ["stern", "deliberate"],
        "notes": "Elena's father",
        "confidence": 0.68
      },
      "voice": {
        "description": "gruff elderly man, slow deliberate speech",
        "qwen_instruct": "A gruff elderly male voice with slow, deliberate pacing."
      }
    }
  ],
  "roles": ["Narrator", "Elena", "Marcus"],
  "types": ["narration", "dialogue", "thought"],
  "script": [[0, 0, 0], [0, 0, 1], [1, 1, 2], [0, 0, 3], [2, 2, 4]]
}
```

Each script row is:

```text
[role_idx, type_idx, sentence_idx]
```

Allowed speech types are:

- `narration`
- `dialogue`
- `thought`

Validation rules:

- Every `sentence_idx` must exist in the sentence artifact.
- Every sentence in the window must appear exactly once.
- Every `role_idx` must point into `roles`.
- Every `type_idx` must point into `types`.
- `roles` must include `Narrator` when narration appears.
- New character names must not collide with existing character names or aliases.
- Output must be valid JSON with no prose wrapper.

On validation failure, v1 retries once with a repair prompt containing the validation errors and the invalid output. If repair fails, v1 saves the failed response for debugging and stops before registry mutation.

## Windowing

Windows are contiguous ranges of saved sentence artifacts. A sentence is atomic and must not be split across LLM or TTS windows.

Window building rule:

```text
if current window plus next sentence fits:
  add the sentence
else:
  close current window
  put the sentence in the next window
```

The LLM window limit is based on configurable token or character estimates. The TTS window limit is based on:

- Maximum 8 roles per window.
- Configurable input size estimate.
- Sentence atomicity.

If one sentence is too large for a configured window, v1 fails with the chapter ID, sentence index, and a short text preview. It does not silently split the sentence.

## Registry

`registry.json` stores durable book and voice state:

```json
{
  "book": {
    "title": "Book Title",
    "slug": "book-title"
  },
  "narrator": {
    "role_id": "narrator",
    "display_name": "Narrator",
    "character_profile": {
      "role": "narrator"
    },
    "voice_identity": {
      "seed": 1001,
      "differentiators": ["calm baseline narrator timbre"]
    },
    "voice_profile": {
      "description": "calm literary narrator, clear pacing",
      "qwen_instruct": "A calm literary narrator voice with clear pacing."
    },
    "voice_config_path": "voices/narrator.qvp"
  },
  "characters": {
    "elena": {
      "role_id": "elena",
      "display_name": "Elena",
      "aliases": [],
      "character_profile": {
        "age_range": "young adult",
        "gender": "female",
        "personality": ["hesitant", "thoughtful"],
        "notes": "main character",
        "confidence": 0.72
      },
      "voice_identity": {
        "seed": 184392,
        "differentiators": ["brighter timbre", "slightly quicker cadence", "lighter resonance"]
      },
      "voice_profile": {
        "description": "young woman, soft, hesitant",
        "qwen_instruct": "A soft young adult female voice, hesitant and thoughtful, with a brighter timbre, slightly quicker cadence, and lighter resonance."
      },
      "voice_config_path": "voices/elena.qvp",
      "first_seen": "chapter_001"
    }
  }
}
```

`role_id` is a stable slug. `display_name` is the user-facing name. Later UI can edit character and voice profiles before regeneration, but v1 has no human review gate.

## Voice Identity And Reproducibility

The text voice description alone is not the reproducibility anchor. For stable voices across chapters, v1 must reuse saved `.qvp` files.

Voice rule:

```text
if voice_config_path exists and the .qvp file is readable:
  load and reuse the .qvp prompt
else:
  generate a voice once from voice_profile plus voice_identity.seed
  save the .qvp file
  update registry.json
```

When a new character's voice profile is identical or too similar to an existing character, v1 adds stable differentiators before voice generation. Differentiators may include:

- Pitch/register.
- Timbre brightness.
- Breathiness or roughness.
- Cadence.
- Accent strength.
- Resonance.
- Energy level.

These differentiators are saved in `voice_identity` and folded into `voice_profile.qwen_instruct`. Two distinct characters with similar age, gender, personality, and role should still sound distinguishable to listeners.

## Qwen TTS Integration

V1 uses a direct Python wrapper around the `qwen_tts` model code from `ComfyUI-Qwen-TTS`.

The wrapper should avoid ComfyUI's internal merge path because v1 needs exact sentence timing metadata. The wrapper should expose:

- Load model.
- Create voice from `qwen_instruct` and seed.
- Load saved `.qvp`.
- Generate a batch of sentence waveforms for a voice prompt.

The ComfyUI project exposes useful reference behavior:

- `RoleBankNode` supports up to 8 roles.
- `DialogueInferenceNode` accepts `RoleName: Text` script lines, batches generation, and can merge output.
- `.qvp` files are used to save and load precomputed voice prompt features.

V1 should preserve the 8-role limit as a TTS windowing constraint.

Voice bootstrap for a missing `.qvp`:

```text
voice_profile.qwen_instruct
  -> generate a short voice-design sample
  -> create a voice clone prompt from that sample
  -> save voices/<role_id>.qvp
  -> save voices/<role_id>.json metadata
  -> update registry voice_config_path
```

## Chapter Audio And Timeline

After annotation validation and voice availability, v1 converts a chapter into ordered sentence jobs:

```json
{
  "sentence_idx": 12,
  "role": "Elena",
  "type": "dialogue",
  "text": "I thought you were gone."
}
```

For each generated sentence waveform, v1 measures duration from sample count and sample rate. It stitches sentence audio with configured pauses and writes both audio and timeline files.

Timeline output:

```json
{
  "chapter": "chapter_001",
  "audio_path": "audio/chapter_001.wav",
  "sample_rate": 24000,
  "sentences": [
    {
      "sentence_idx": 0,
      "role": "Narrator",
      "type": "narration",
      "start_ms": 0,
      "end_ms": 2840
    }
  ]
}
```

This supports sentence-level highlighting in the future reader UI. V1 does not attempt word-level alignment.

## Configuration

Configuration should cover:

- Anthropic API key environment variable name: `ANTHROPIC_API_KEY`.
- Anthropic model: default Haiku, overrideable.
- Anthropic max retries: default 1 repair retry.
- Qwen model choice: default `1.7B`, overrideable.
- Qwen device: `auto`, `cuda`, `xpu`, `mps`, or `cpu`.
- Qwen precision: default `bf16`.
- Qwen attention: default `auto`.
- Max LLM window size.
- Max TTS window input size.
- Max roles per TTS window: 8.
- Pause durations.
- Output audio format: WAV first.
- Book root path.

## Failure Behavior

The pipeline should fail early and clearly for:

- Missing `ANTHROPIC_API_KEY`.
- Missing Qwen runtime dependencies.
- Missing Qwen model files.
- Low-confidence chapter split.
- Bad LLM JSON after repair retry.
- Annotation with missing or duplicate sentence IDs.
- New character collision with existing aliases.
- Single sentence too large for a configured window.
- Voice generation failure.
- Unreadable `.qvp` file.

Registry writes should be atomic. V1 should not leave a half-updated registry if voice creation fails.

## Testing And Verification

Unit tests:

- Chapter splitter confidence behavior.
- Sentence artifact shape and stable indexes.
- Annotation validator accepts valid compact JSON.
- Annotation validator rejects missing, duplicate, or out-of-range sentence IDs.
- Character collision detection.
- Window builder preserves sentence atomicity.
- TTS role windowing respects the 8-role limit.
- Timeline duration math from waveform sample counts.
- Atomic registry update behavior.

Fixture tests:

- Fake Anthropic adapter returns known annotations.
- Fake Qwen adapter returns deterministic waveforms.
- End-to-end fixture creates annotation, synthetic audio, and timeline from a tiny chapter.

Manual smoke test:

- Requires real Anthropic API access and local Qwen model setup.
- Processes a short chapter with narrator plus two characters.
- Confirms `annotation.json`, `registry.json`, `.qvp` files, `chapter_001.wav`, and `chapter_001.timeline.json` are written.

## References

- Anthropic model documentation: https://docs.anthropic.com/en/docs/about-claude/models/overview
- ComfyUI-Qwen-TTS repository: https://github.com/flybirdxx/ComfyUI-Qwen-TTS
- Inspected Qwen reference classes: `RoleBankNode`, `DialogueInferenceNode`, `SaveVoiceNode`, `LoadSpeakerNode`, and `qwen_tts.inference.qwen3_tts_model.Qwen3TTSModel`.
