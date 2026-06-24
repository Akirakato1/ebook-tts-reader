# Read-Along Web UI Design

## Goal

Add a second read-along UI surface that runs like a local notebook: a terminal command starts a local HTTP server, opens the default browser, and serves a clean web reader backed by the same read-along pipeline used by the desktop UI.

## Scope

The web app is local-only. It does not replace the Tk UI. It reuses `PrototypeUiController`, `ReadAlongSession`, read-along units, chapter text, settings, and voice preparation. It does not introduce a web framework dependency.

## User Experience

The first screen is the read-along reader, not a marketing page. The layout has a narrow contents/sidebar and a natural book text pane. The chapter text is rendered as continuous reading text; sentence and quote units are highlighted inline. The user can select a starting unit by clicking the text.

Controls include:
- refresh chapters
- process book
- build chapter units
- playback speed
- generation mode
- buffer limit
- narrator voice type
- start session
- end session

Settings lock when a session starts. Changing speed, buffer size, narrator type, or generation mode requires ending the session first.

## Server

Create `ebook_tts_pipeline.ui.web_app`. The server uses Python stdlib `http.server` and `ThreadingHTTPServer`. A new console script named `ebook-tts-readalong-web` starts it.

Startup behavior:
- accept `--book-root`, `--host`, `--port`, `--fake-tts`, and `--no-open`
- use port `0` or an occupied-port fallback to select an available port
- print the URL
- open the browser unless `--no-open` is set

## API

The browser UI calls JSON endpoints:
- `GET /api/state`: current chapters and settings
- `GET /api/chapter/<chapter>`: chapter text and read-along units
- `POST /api/process-book`: run global registry and quote attribution for all chapters
- `POST /api/build-units`: build units for one chapter
- `POST /api/session/start`: lock settings, create session, fill initial buffer
- `POST /api/session/advance`: consume the current unit and refill within the buffer limit
- `POST /api/session/end`: stop generation and delete temp audio
- `GET /api/session/<session_id>/audio/<unit_id>.wav`: serve generated temp audio

Each audio response is served from the session directory only. The server never serves arbitrary filesystem paths.

## Data Flow

The web server owns a `PrototypeUiController` and a single active session per process. Session start creates `ReadAlongSession`, fills the initial buffer, returns ready unit IDs and audio URLs. The browser plays the first ready audio, highlights the current unit, then calls `advance` after playback ends. The server consumes the previous audio, refills one slot, and returns the new ready queue.

## Error Handling

API errors return JSON with `ok: false` and a human-readable `error`. Pipeline failures include the normal debug log path text from existing controller/pipeline errors when available. `process-book` and audio generation requests are synchronous for this first web version.

## Testing

Tests use fake TTS and temporary book roots. They verify:
- the server returns the HTML shell
- state includes chapters and settings
- chapter response includes continuous text and units
- session start respects buffer limit and returns audio URLs
- audio endpoint serves WAV bytes for ready units
- advance consumes/refills without exceeding buffer limit
- end deletes session audio

Chapter 15 validation reuses the existing False Witness chapter 15 artifacts without modifying real voice prompts by copying required files to a temp root and using fake TTS.
