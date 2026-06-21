# Tkinter Prototype UI Design

## Goal

Build a simple local Python UI for the ebook-to-audiobook pipeline so the user can load an EPUB, initialize chapters/sentence segments/registry, annotate chapters, edit the registry before script generation, generate Qwen/TTS scripts, synthesize audio, and open completed chapter audio.
The prototype also keeps a local `books/library.json` index so the user can switch between previously loaded books.

## Approach

This is a prototype UI, not the final reader interface. The implementation uses Tkinter because it ships with Python and can call the existing pipeline directly. The important boundary is a testable `ui.controller` module with no Tkinter dependency, so a future Electron UI can reuse the same workflow decisions or call equivalent CLI/API methods.

## Workflow

1. User selects an EPUB and book output folder.
2. The UI extracts chapters, sentence-segments each chapter, initializes `registry.json`, and shows a chapter table.
3. The UI registers the book in `books/library.json`; selecting a book from the book list switches the active book root and refreshes chapters/registry.
4. Each chapter button reflects artifact state:
   - Gray: chapter and sentence segments exist, annotation not generated.
   - Green: annotation exists and registry has been updated; clicking builds `.tts_script.json` and `.qwen_script.txt` from the current edited registry.
   - Blue: TTS/Qwen scripts exist; clicking generates chapter audio.
   - Yellow: audio exists; clicking opens the audio file.
5. The registry panel can be toggled open or closed at any time.
6. The registry panel edits raw pretty JSON for `registry.json` and validates JSON before save.

## Constraints

- Long operations run in a background thread so the UI remains responsive.
- Registry edits are saved before script generation if the user clicks save.
- Audio generation uses the current Qwen adapter path by default, with a fake-TTS option for testing/prototyping.
- The prototype does not include ebook reading, sentence highlighting playback, or rich character form editing. Raw JSON editing is enough for this pass.

## Testing

Tests focus on the controller, not Tkinter widgets:

- Detect chapter artifact states from files.
- Initialize a book from an EPUB using injectable pipeline functions.
- Validate and save registry JSON.
- Advance the chapter action from annotate to script generation to audio generation/open.
