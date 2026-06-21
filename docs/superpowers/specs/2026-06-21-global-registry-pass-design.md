# Global Registry Pass Design

## Goal

Add a book-level character registry pass before chapter annotation so repeated characters such as Akari are represented once with aliases, instead of being rediscovered independently in later chapters.

## Pipeline Shape

After EPUB extraction and deterministic sentence segmentation, the app can run a global registry pass. The pass sends compact whole-chapter text windows to the configured Anthropic model, defaults to the existing Haiku model, and asks only for canonical character profiles, aliases, age-stage variants, and chapter evidence. The default global-registry window is 130,000 chapter-text characters, configurable with `EBOOK_TTS_GLOBAL_REGISTRY_WINDOW_CHARS`. It does not produce sentence-level annotation.

Each global-registry window receives a minimal summary of the current saved registry. Each existing character summary contains only `name`, `age_stage`, and `description`. The saved `registry.json` remains rich enough for UI editing and voice caching, but the prompt omits ids, aliases, voice variants, Qwen instructions, hashes, old evidence, and narrative notes. After a successful window, the app immediately merges the returned new characters into `registry.json`; the next window sees updated character summaries. The prompt treats the existing registry as authoritative, instructs the model not to recreate summarized characters, and requires the response to contain new characters only.

The registry remains user-editable in the prototype UI. Chapter annotation then runs against this locked registry and should not automatically create new registry records. If the model sees an unknown speaker during chapter annotation, it returns `proposed_new_characters`; the app records them in the annotation JSON for review but does not add them to `registry.json`.

## UI Flow

The prototype UI gets a `Build Global Registry` button. Loading a book still extracts chapters, writes sentence segment files, initializes an empty registry, and writes the table of contents. The user can then build or rebuild the global registry, inspect/edit it in the registry panel, and click chapter buttons to annotate against the locked registry.

## Data Shape

The global registry LLM response uses:

- `characters`: list of `{name, profile, evidence}`
- `evidence`: compact chapter/sentence references and short notes used for debugging identity merges

The saved registry keeps the existing character record shape so downstream voice caching and script generation continue to work.

## Error Handling

Existing debug logging captures global-registry prompt/output failures. Failed global windows should not partially corrupt the registry. Successful windows are merged through the existing registry manager, which already rejects collisions; the global pass should add new records while avoiding existing summarized characters.

## Testing

Tests cover global prompt rendering, global service validation, registry merge behavior, pipeline global registry generation, locked chapter annotation behavior, and the UI controller button workflow.
