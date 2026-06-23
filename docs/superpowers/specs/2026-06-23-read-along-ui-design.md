# Read-Along UI Design

Date: 2026-06-23

## Goal

Build a read-along mode for the existing ebook TTS app. The mode should share the front half of the audiobook generation pipeline: book selection/import, global registry creation, sentence/quote extraction, quote attribution annotation, local temporary speakers, and voice preparation.

The reader experience should look like a normal ebook app. Text is displayed page by page with natural paragraph formatting. The current sentence or quote is highlighted inline inside the page text, not shown as a list of segmented boxes. Segment/unit rows remain internal data for audio generation, highlighting, logging, and debugging.

## Approved UX

Add a Read Along tab to the existing Tkinter app rather than creating a separate app. The tab reuses the current book library list and chapter table of contents.

The reader surface has three main areas:

- Left: book list and table of contents.
- Center: paginated ebook-style text view with inline highlights.
- Right: session controls, locked settings, buffer state, and optional debug access.

When a user loads or processes a book, the UI shows that the book is processing. Processing runs or reuses the global registry pass, then runs or reuses quote attribution annotation for every chapter. Once chapters are processed, the app can build read-along unit metadata for each chapter.

When reading, the user can navigate by table of contents and page. Clicking the page text selects the nearest sentence/quote unit as the start point. The user then starts a session from that selection.

## Locked Session Settings

The following settings are editable before a read-along session starts and locked while a session is active:

- Playback speed.
- Generation speed mode/slider.
- Narrator voice type.
- Buffer limit, defaulting to 2 sentence/quote units.

To change locked settings, the user must end the session first. This keeps audio generation, playback speed, buffer accounting, and voice selection from changing mid-stream.

Narrator voice type should update the narrator profile before voice preparation. The first implementation should support a small explicit set, such as male narrator, female narrator, and custom/current narrator profile. The narrator voice record must remain a stable registry voice keyed by role_id `narrator`.

## Read-Along Units

A read-along unit is one generated audio/highlight target. It can be narrator text, a spoken quote, a narrator quote, or a dialogue attribution/tag span.

Each unit must include:

- Chapter id.
- Unit id in reading order.
- Display text.
- Source start/end character offsets in the chapter text.
- Role display name and stable role_id.
- Speech type.
- Voice config path.
- Optional quote id, sentence id, and local speaker metadata.

The key change from the current TTS script artifact is preserving source offsets. Current quote extraction already provides offsets for quotes and narrator spans, but the TTS script currently drops those offsets when it creates TTS jobs. Read-along metadata should preserve them so inline highlighting can be exact and does not rely on fuzzy string matching.

## Highlighting And Pagination

The chapter text is rendered as paragraphs in a page-like text view. Active and queued units are represented with text tags over the original rendered text:

- Current unit: primary highlight.
- Queued/generated unit: subtle secondary highlight if useful.
- Selection/start unit: selection tag before playback begins.

The visible page should auto-advance or scroll when playback reaches a unit that is outside the current page. Page navigation should remain available when no session is active. During playback, manual navigation is allowed for browsing, but the active highlight should be able to snap back to the current unit.

The unit list/debug transcript is not the primary UI. It may be exposed behind a debug toggle for investigating role assignment, voice paths, generation speed, and unit boundaries.

## Buffer Runtime

A read-along session creates a temporary session audio directory. Generated audio is written as temporary per-unit WAV files or in-memory audio chunks that are deleted when consumed.

The default buffer limit is 2 units. The runtime must never have more than the configured number of ready/generated future units. It may request a 2-unit model call when two buffer slots are open, but it must not generate beyond the buffer limit.

Playback flow:

1. User starts from a selected unit.
2. UI shows "building buffer".
3. Runtime fills up to the buffer limit.
4. Playback starts once the initial buffer is ready.
5. While the current unit plays, generation tries to fill the next open buffer slot.
6. Consumed audio is deleted.
7. At chapter end or End Session, playback stops, generation stops, temp audio is deleted, and settings unlock.

The runtime must tolerate under-runs. If playback reaches the end of ready audio before the next unit is generated, the UI shows that it is rebuilding the buffer and resumes once ready.

## Performance Findings From Chapter 15

Chapter 15 of False Witness was benchmarked with the current Qwen setup at playback speed 1.0.

Single-unit mode:

- 30 units measured.
- Aggregate realtime factor: 2.66x slower than playback.
- Median realtime factor: 2.61x.
- Realtime-compatible units: 0/30.

Two-unit model-call mode:

- 8 pairs measured.
- Aggregate realtime factor: 1.98x slower than playback.
- Mixed-role pairs improved to roughly 1.38x to 1.88x.
- Realtime-compatible pairs: 0/8.

Conclusion: strict one-unit generation is useful for debugging voice assignment but is not fast enough for seamless playback on the current runtime. Two-unit calls help, especially mixed-role calls, but a two-unit buffer still cannot honestly promise seamless 1.0x playback. The UI must surface buffer-building pauses and log realtime factor until runtime optimization closes the gap.

## Generation Speed Setting

Generation speed mode controls runtime behavior within the buffer limit:

- Precise/debug: generate one unit per call. Best for voice attribution debugging.
- Balanced: use up to the available buffer slots, but prefer safer grouping.
- Fast: use up to the available buffer slots and allow mixed-role multi-role calls when safe.

The first implementation should store the selected mode on the session log and use it to decide one-unit versus up-to-two-unit model calls. It should not change the default buffer limit of 2.

## Logging

Every generated unit or buffer call should be logged to JSONL with:

- Chapter and unit id.
- Role, role_id, speech type, and voice_config_path.
- Source offsets and text character count.
- Generation seconds.
- WAV/write seconds if written to disk.
- Raw audio duration.
- Playback duration after speed.
- Realtime factor.
- Buffer wait/under-run state.
- Success or error.

This log is part of the product, not just temporary debugging. It is how we determine whether read-along can become seamless and how we catch voice assignment regressions.

## Error Handling

If book processing fails, show the UI pipeline error dialog and write the existing failure log.

If a chapter lacks registry or annotation artifacts, the Read Along tab should offer to process the book or chapter before starting playback.

If a voice file is missing, the app should prepare voices before the session starts. If voice preparation fails, the session should not start.

If TTS generation fails during a session, stop playback, clear temporary audio, unlock settings, and show the failing unit role_id and log path.

If the user ends a session, the app should cancel pending generation, stop playback, delete temp audio, and leave persistent registry/annotation artifacts intact.

## Implementation Boundaries

New read-along behavior should be split into focused modules rather than growing the existing Tk file too much:

- A read-along unit builder that converts chapter text plus quote attribution into offset-preserving units.
- A read-along session/runtime that owns buffer state, temp audio, playback, cancellation, and timing logs.
- Tk UI additions for the Read Along tab, natural page text tags, controls, and session state.
- Controller methods that expose book processing and read-along artifacts to the UI.

Existing audiobook generation should remain available. The read-along path must not delete or rewrite full-chapter audiobook outputs unless the user explicitly uses audiobook actions.

## Testing

Automated tests should cover:

- Building read-along units from quote attribution while preserving source offsets and role_id/voice path data.
- Mapping a clicked text offset to the correct unit.
- Enforcing the buffer limit of 2.
- Locking and unlocking session settings.
- Cleaning temporary audio on end session and chapter end.
- Logging generation/playback timing and realtime factor.
- Controller behavior for processing a book and exposing chapter/page/unit data.

Manual or integration validation should use False Witness chapter 15:

- Open the Read Along tab.
- Load/select the False Witness book.
- Process or reuse chapter 15 artifacts.
- Display chapter 15 as natural page text.
- Select a start unit from page text.
- Start a read-along session with buffer limit 2.
- Confirm inline highlight moves with playback.
- Confirm generation timing logs are written.
- Confirm temp audio is deleted on End Session.

