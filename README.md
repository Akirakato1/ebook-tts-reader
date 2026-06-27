# Ebook TTS Reader

Local ebook read-along and audiobook generation tooling. The current production focus is
the `readalongweb` app: a notebook-style local web UI for loading EPUBs, building a
global character registry, annotating quotes chapter by chapter, generating reusable
character voice profiles, and playing a highlighted read-along session with a time-based
audio buffer. The same web app now also includes an audiobook screen for persistent
per-chapter WAV generation and playback.

The older long-form audiobook pipeline is still in the repository, but the active path is
the read-along web app and its WSL/Qwen read-along TTS stack.

## Current Status

- `readalongweb` launches a local browser UI from a library folder, like starting a
  notebook server from the directory you want to work in.
- Books move through explicit stages: add EPUB, initialize chapters/segments, build the
  global registry, annotate every chapter, review registry voices, generate voices, then
  open the book for read-along playback.
- `Review Voices` is the registry editing surface. `Save All` only writes profile
  metadata and invalidates stale voice readiness; it does not load TTS or regenerate
  samples. The library shows `Generate Voices` whenever any global registry voice is
  missing, missing its sample, or has a stale `voice_config_hash`.
- The book page renders natural chapter text and highlights the current sentence or quote
  unit during playback.
- Start-session settings lock the narrator type, playback speed, and buffer window for
  that session. End the session before changing those settings.
- Voice-ready books can be exported as portable ReadAlong zip packages and imported into
  another library folder without copying logs or generated playback buffers.
- Voice-ready books can generate audiobook chapters from the book page. This path writes
  persistent WAV/timeline files under `audiobook/` and is optimized for whole-chapter
  throughput with larger TTS windows, not live segment latency.
- The default read-along TTS backend is the WSL vLLM-Omni/Qwen3-TTS 12 Hz stack found in
  the benchmark notes.

## Quick Start

From the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
readalongweb books
```

`readalongweb` opens the local web UI automatically. Use `--no-open` when you want to
start the server without opening a browser:

```powershell
readalongweb books --no-open
```

Useful launch options:

```powershell
readalongweb                         # use the current directory as the library root
readalongweb books                   # use ./books as the library root
readalongweb --root books            # same as above
readalongweb --book-root books/demo  # open a single book root directly
readalongweb --fake-tts              # UI/session testing without the real TTS backend
```

After `pip install -e .`, the `readalongweb` command is available from any terminal that
uses this Python environment.

## Book Flow

1. Add Book: select an EPUB, title, author, and optional slug. The app creates a book
   folder under the current library root.
2. Initialize Book: extracts chapter text and creates the initial sentence/read-along
   structure.
3. Build Registry: runs the LLM global registry pass for important recurring characters.
4. Annotate Book: annotates all chapters, writes quote attribution, chapter-local
   registries, and read-along units. The button can resume from a failed or unfinished
   chapter.
5. Review Voices: opens the editable global registry character profiles. Use the single
   `Save All` button to persist edits; custom accents are allowed, and blank accents can
   be inferred from ethnicity/background when possible.
6. Generate Voices: creates or refreshes QVP voice files and intro WAV samples for global
   registry characters only. This button appears when not all global voices are current,
   including after a profile edit changes the hash.
7. Share Zip, optional: exports a portable `.readalong.zip` package for another
   `readalongweb` library.
8. Open Book: click the book title area once the book is voice-ready.
9. Start Session: choose the start unit, narrator voice type, playback speed, and buffer
   seconds. The session builds a WAV time buffer before playback begins.
10. Audiobook, optional: open the Audiobook screen, choose chapters, and generate
    persistent chapter WAVs. Audiobook mode has its own narrator profile separate from
    the read-along session narrator. Generated chapters can be played in the embedded
    media player with speed and continue-to-next-ready-chapter controls.

Generated book data lives inside each book folder:

```text
<library-root>/<book-slug>/
  chapters/
  annotations/
  read_along/
  audiobook/
    narrator_profile.json
    settings.json
    chapter_001.wav
    chapter_001.timeline.json
    manifest.json
  temp_registries/
  voices/
    *.qvp
    _samples/*.wav
```

Books, models, generated audio, logs, and test output folders are intentionally local
runtime data and are not meant to be committed.

## Sharing Books

The library view exposes two portable-book actions:

- `Share Zip`: available after a book reaches `Voices ready`. It downloads a
  `<book-slug>.readalong.zip` package.
- `Import Zip`: imports a package produced by `Share Zip` into the current library root
  and creates a normal voice-ready book entry.

The share package includes only the files needed to open and run read-along sessions:

```text
readalong_book.json
registry.json
toc.json
chapters/
sentence_segments/
annotations/
read_along/*.units.json
read_along/settings.json
read_along/narrator_profile.json
audiobook/settings.json
audiobook/narrator_profile.json
temp_registries/
voices/*.qvp
voices/_samples/*.wav
```

The package intentionally excludes the source EPUB, logs, `read_along_sessions/`,
temporary playback WAVs, runtime narrator QVPs, and chapter-local temp speaker QVPs.
Imported books regenerate narrator/session voices as needed from the receiving user's
session settings and local TTS stack.

## Voice Lifecycle

There are three voice categories:

- Global registry characters: generated by the `Generate Voices` button before the book
  is opened for TTS. Each character gets a `.qvp` file and a short
  `Hello, my name is ...` sample under `voices/_samples/`.
- Registry samples are the actual VoiceDesign reference utterances used to create the
  QVP files. They are not generated by a second voice-clone TTS pass. If an old QVP
  exists without its matching sample, `Generate Voices` regenerates the QVP and sample
  together so the preview remains truthful.
- Global registry voice readiness is hash-based. A character is ready only when its QVP
  exists, its intro sample exists, and `voice_config_hash` matches the current compact
  voice profile. A stale manifest flag cannot override this check.
- Registry profile saves are intentionally metadata-only. They return quickly, mark
  changed voices stale, and let the separate `Generate Voices` job rebuild QVP/sample
  assets with progress tracking.
- Accent labels are expanded into explicit pronunciation constraints before QVP
  generation. For example, `British`, `Received Pronunciation`, `Yorkshire`, and
  `French` become concrete guidance about rhoticity, vowel color, consonants, and
  regional drift rather than a bare accent name. Registry samples use a longer phrase
  with accent-revealing words so the preview is easier to judge.
- Narrator: generated or refreshed when a read-along session starts, based on the
  session narrator setting. Normal narration and `narrator_quote` units share this
  one narrator voice.
- Audiobook narrator: configured separately on the Audiobook screen and generated when
  audiobook chapters are rendered. It can intentionally differ from the read-along
  narrator.
- Chapter-local/temp speakers: generated only when needed during a read-along session.
  They are not part of the pre-session global voice readiness count.

If the terminal shows local/temp speakers being generated during `Generate Voices`, that
is a bug. If it shows model downloads during a normal run, the local model path or backend
configuration is probably wrong.

## TTS Stack

Default environment-driven settings live in
`src/ebook_tts_pipeline/config.py`.

Important defaults:

```powershell
$env:EBOOK_TTS_QWEN_MODEL_ROOT = "models/qwen-tts"
$env:EBOOK_TTS_VOICE_ASSET_BACKEND = "wsl"
$env:EBOOK_TTS_READ_ALONG_BACKEND = "wsl-vllm-omni"
$env:EBOOK_TTS_WSL_DISTRO = "Ubuntu-24.04"
$env:EBOOK_TTS_WSL_PYTHON = "/opt/ebook-tts-venv/bin/python"
$env:EBOOK_TTS_VLLM_OMNI_WSL_PYTHON = "/opt/ebook-vllm-omni-venv/bin/python"
```

Expected local Qwen model layout:

```text
models/qwen-tts/
  Qwen3-TTS-12Hz-1.7B-Base/
  Qwen3-TTS-12Hz-1.7B-VoiceDesign/
  Qwen3-TTS-Tokenizer-12Hz/
```

The read-along backend currently uses the vLLM-Omni 12 Hz profile:

```text
scripts/vllm_omni_qwen3_tts_16gb_len8192_seq2_s0util026.yaml
```

That profile was chosen because benchmarks showed it could generate faster than playback
with stable read-along buffering on the test laptop GPU. It should be treated as a
16 GB NVIDIA CUDA requirement: confirmed benchmark rows sat around 11.9 GB peak, while
live vLLM sessions can reserve roughly 12-14 GB. The lower ~10.7 GB experimental profile
was not stable enough to expose as a production setting.

The generation selector in the web UI controls batching behavior, not the resident vLLM
memory profile:

- `Balanced (12-14 GB, RTF ~0.15)`: default. Generates up to two read-along units per
  request while still returning per-unit audio for highlighting. Benchmarked smooth
  ceiling was about 6.6x at 1.0x playback.
- `Precise (12-14 GB, RTF ~0.25)`: one unit per request. Useful for debugging voice
  attribution or fidelity, with less speed headroom.
- `Burst (12-14 GB, fastest fill)`: uses the same resident VRAM profile but allows larger
  queue fills when buffer time allows.

Audiobook generation uses the same 12 Hz accelerated stack but different batching:

- Read-along sessions generate a time-based buffer of small units so text highlighting
  can stay exact and responsive.
- Audiobook chapters generate larger character/role windows through
  `ChapterAudioBuilder.build_chapter_audio_from_windows()`, then stitch the chunks into
  persistent per-chapter WAV files. Audiobook narration uses `audiobook/narrator_profile.json`,
  not `read_along/narrator_profile.json`. This is the path to use when total chapter
  generation speed matters more than immediate playback.
- The exposed audiobook model profile is 12 Hz only for now. A 25 Hz smoke test could not
  resolve a public/local usable Qwen3-TTS 25 Hz repo, so the UI intentionally avoids
  presenting a 25 Hz option until that stack is confirmed.

An 8 GB GPU can use slower native/WSL paths for experimentation or offline work, but the
current smooth local read-along target requires the 16 GB accelerated vLLM stack. See:

- `docs/benchmarks/readalong_tts_vllm_omni_experiment_summary.txt`
- `docs/benchmarks/2026-06-23-chapter-015-window-sweep.md`
- `docs/benchmarks/2026-06-23-chapter-15-wsl-flashattention.md`
- `docs/runbooks/manual-tts-stack.txt`

Runtime diagnostics are printed with an `[ebook-tts]` prefix. The most useful lines for
stack verification are `voice_asset_pipeline`, `build_tts_adapter`,
`wsl_qwen_adapter_config`, `vllm_omni_adapter_config`, and read-along session/buffer
events.

## LLM Configuration

Annotation and registry construction require Anthropic access:

```powershell
$env:ANTHROPIC_API_KEY = "..."
$env:EBOOK_TTS_ANTHROPIC_MODEL = "claude-sonnet-4-6"
```

The annotator is quote-centric. It receives chapter quote context, the global registry,
and instructions for chapter-local speakers. Ambient/non-profile speech can be labeled as
`narrator_quote`, which is handled by the same session narrator voice during read-along
generation.

## Development

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Useful docs:

- `docs/pipeline-map.md` maps the older audiobook pipeline.
- `docs/benchmarks/` contains the TTS speed and VRAM experiments.
- `config.example.toml` shows the main pipeline configuration knobs.

