# Global Registry Pass Design

## Goal

Add a book-level character registry pass before chapter annotation so repeated characters such as Akari are represented once with aliases, instead of being rediscovered independently in later chapters.

## Pipeline Shape

After EPUB extraction and deterministic sentence segmentation, the app can run a global registry pass. The pass sends compact whole-chapter text windows to the configured Anthropic model, defaults to the existing Haiku model, and asks only for canonical character identity facts and age-stage variants. The default global-registry window is 130,000 chapter-text characters, configurable with `EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS`. It does not produce sentence-level annotation.

Each global-registry window receives a minimal summary of the current saved registry. Each existing character summary contains only `name`, `age_stage`, `gender`, `race_or_accent`, `occupation`, and `personality_type`. The LLM returns the same compact fields for new characters and meaningful updates only. The app expands those compact deltas into saved registry records, then the next window sees updated summaries. The saved `registry.json` remains rich enough for UI editing and voice caching, but the prompt omits ids, aliases, voice variants, Qwen instructions, hashes, old provenance, and narrative notes. The prompt treats the existing registry as authoritative, instructs the model not to recreate summarized characters, and allows updates only when a chunk adds or corrects one of those key identity facts.

When a compact delta has the same `name` and the same known `age_stage` as an existing record, it updates missing facts on that record. When it has the same `name` but a different known `age_stage`, the app creates a separate voice profile such as `callie_teen` next to `callie_adult`.

The registry remains user-editable in the prototype UI. Chapter annotation then runs against this locked registry and should not automatically create new registry records. The annotation prompt receives compact character summaries with names, aliases, age stage, gender, race/accent, occupation, and personality traits, while omitting voice variants, Qwen instructions, seeds, hashes, and `.qvp` cache paths. If the model sees an unknown speaker during chapter annotation, it returns `proposed_new_characters`; the app records them in the annotation JSON for review but does not add them to `registry.json`.

## UI Flow

The prototype UI gets a `Build Global Registry` button. Loading a book still extracts chapters, writes sentence segment files, initializes an empty registry, and writes the table of contents. The user can then build or rebuild the global registry, inspect/edit it in the registry panel, and click chapter buttons to annotate against the locked registry.

## Data Shape

The global registry LLM response uses:

- `characters`: list of `{name, age_stage, gender, race_or_accent, occupation, personality_type}`
- `name`: stable display name for the person in that life-stage row
- `age_stage`: one of `child`, `teen`, `adult`, `elder`, or `unknown`
- `race_or_accent`: compact combined string such as `Japanese; Tokyo accent`, or `unknown`
- `personality_type`: short comma-separated voice-casting traits

The saved character registry keeps identity fields, aliases, and voice cache fields only. Deprecated plot/provenance fields such as `timeline`, `same_person_as`, `character_profile`, `narrative_notes`, `first_seen`, and `global_evidence` are pruned on save or merge.

## Error Handling

Existing debug logging captures global-registry prompt/output failures. Failed global windows should not partially corrupt the registry. Successful windows are merged through the existing registry manager, which already rejects collisions; the global pass should add new records or update existing records only with newly discovered key facts.

## Testing

Tests cover global prompt rendering, global service validation, registry merge behavior, pipeline global registry generation, locked chapter annotation behavior, and the UI controller button workflow.
