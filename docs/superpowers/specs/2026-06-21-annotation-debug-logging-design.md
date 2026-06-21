# Annotation Debug Logging Design

## Goal

Save local debug evidence automatically whenever annotation or UI pipeline work fails, so failures can be inspected without paying for another blind Anthropic retry.

## Scope

The first implementation focuses on the prototype pipeline and UI. It logs annotation model-output failures, annotation validation failures, malformed annotation payloads, and uncaught UI background exceptions. It does not add a log viewer yet.

## Log Location

Logs are written as timestamped JSON files under `logs/annotation_failures/`. The directory is already ignored by git through the existing `logs/` ignore rule.

## Log Contents

Annotation logs include book root, chapter, event type, sentence index range, sentence count, prompt text, parsed payload when available, raw model text when available, exception type, exception message, and traceback. Full prompts are intentionally stored locally because they are the most useful evidence for debugging paid API failures.

UI logs include action label, book root, exception type, message, and traceback. If an exception already has an annotation debug log path, the UI reports that path instead of writing a duplicate generic log.

API keys are not logged. The sanitizer redacts Anthropic-style API keys and bearer tokens if they accidentally appear in exception text or payloads.

## User Experience

When a background UI action fails, the error popup includes the debug log path. For annotation failures that split and recover, logs may exist even when the UI action completes; this is intentional because the failed paid call is still useful evidence.

## Testing

Tests cover direct failure-log writing, annotation model-output failure logging, validation failure logging before repair, and UI error-message formatting for exceptions with saved debug log paths.
