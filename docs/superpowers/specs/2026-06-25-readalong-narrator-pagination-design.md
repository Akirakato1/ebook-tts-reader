# Read-Along Narrator Profile And Pagination Design

Date: 2026-06-25

## Goal

Improve the read-along experience in two linked areas:

- Replace the shallow narrator `male/female/current` setting with a full editable narrator voice profile.
- Replace the long scrolling chapter view with ebook-style pages that can switch between a two-page spread and a single-page view.

This applies to the `readalongweb` app. The older audiobook generation UI is out of scope.

## Narrator Profile

Use one editable narrator profile per book for this phase. Do not add multiple saved narrator presets yet.

The profile uses the same user-facing schema style as global character voice profiles:

- display name or label
- age stage
- gender
- personality / delivery traits
- race or ethnicity, when relevant
- accent, chosen from the same Qwen-supported/detected option list used for characters
- occupation or role framing, defaulting to audiobook narrator

The narrator profile is not a global registry character and must not appear in the global character voice count. Store it at:

```text
read_along/narrator_profile.json
```

If that file is missing, migrate the initial profile from the existing top-level `registry.narrator` field when available. After migration, `read_along/narrator_profile.json` is the source of truth.

## Narrator QVP Lifecycle

Narrator QVP generation happens at read-along session start, not during `Generate Voices`.

Session start must:

1. Read the current narrator profile.
2. Build the compact Qwen voice profile from that data.
3. Compute the same style of voice profile hash used for character QVP caching.
4. Reuse the cached narrator QVP if the hash matches.
5. Generate a new narrator QVP only when the profile hash changed or the cached file is missing.
6. Build the functional narrator profile as a deterministic derivative of the narrator profile.
7. Apply the same hash/cache rule to the functional narrator QVP.

Cache layout:

```text
voices/_narrator/
  <narrator-profile-hash>/
    narrator.qvp
    functional_narrator.qvp
```

This keeps narrator voices out of global character readiness, avoids stale `voices/narrator.qvp` ambiguity, and still avoids unnecessary regeneration when settings are unchanged.

## Session Settings UI

Replace the current narrator select with a compact narrator profile control:

- A summary label such as `Narrator: Warm adult female, neutral American accent`.
- An `Edit Narrator` button that opens the same clean profile editor style used by Review Voices.
- Profile edits are disabled while a read-along session is active.
- Saving the narrator profile updates the profile hash source but does not immediately generate QVPs.
- Start Session locks the narrator profile for that session.

The playback speed, generation mode, buffer seconds, and max unit controls remain session settings.

## Reader Pagination

The reader should no longer render the chapter as one continuous scroll.

The page layout has two modes:

- Sidebar hidden: two-page spread, like an open ebook.
- Sidebar shown: single-page reader with the chapter list visible.

Add a sidebar toggle button. The current chapter list can slide/collapse in and out. The layout mode follows the sidebar state rather than being a separate user setting.

Manual navigation:

- Left/right buttons appear near the page corners.
- Left/right arrow keys turn pages when the read-along session is not active and focus is not inside an input/select/button.
- In two-page spread mode, page navigation advances by a spread.
- In single-page mode, page navigation advances by one page.

During an active session, playback owns page movement. Manual page turns should not fight playback.

## Active Segment Rule

The page containing the active segment must be visible after any automatic layout transition.

The active segment is:

- before session start: the red outlined selected segment
- during session playback: the current highlighted segment being read

When the sidebar is toggled, the chapter changes layout, or playback advances to a segment on another page, the reader must move to the page or spread containing that active segment.

Manual page navigation before session start is allowed to move away from the selected segment. Clicking a segment on the visible page updates the selected segment and last-read position.

## Pagination Implementation

Use browser-measured logical pages rather than a fixed character count.

Implementation approach:

1. Render the chapter into an offscreen page measurer using the same fonts, width, line height, and padding as the visible page.
2. Add plain text and unit spans in source order until the page overflows.
3. Back off to the last safe unit/text boundary and start the next page.
4. Store each logical page as a list of text slices and read-along unit spans.
5. Render either one logical page or two adjacent logical pages depending on sidebar state.
6. Maintain a `unit_id -> page_index` map for highlight, click selection, and auto-turn.

This keeps natural book-like pages while preserving exact sentence/quote spans for highlighting and click-to-start selection.

## Error Handling

- If a chapter has no read-along units, keep Start Session disabled and show the existing processing message.
- If pagination cannot measure pages yet because the page container is hidden or has zero dimensions, delay pagination until the reader is visible.
- If a session-start TTS error occurs, show the full error in a visible status area or modal instead of silently resetting the Start Session button.
- If narrator QVP generation fails, the session should not start.

## Testing

Add tests for:

- Narrator settings save/load with profile fields instead of only `narrator_voice_type`.
- Narrator QVP cache reuse when the narrator profile hash is unchanged.
- Narrator QVP regeneration when an editable narrator profile field changes.
- Functional narrator cache uses a derivative hash and does not collide with normal narrator.
- `Generate Voices` still counts only global registry characters and does not prepare narrator/local voices.
- Web shell contains sidebar toggle, page navigation buttons, narrator profile editor entry point, and no old `male/female/current` narrator select.
- Page-rendering JavaScript exposes the anchor behavior: sidebar toggle and playback highlight move to the page containing the selected/current unit.
