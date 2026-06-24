# Readalongweb Book Lifecycle Design

## Goal

Make the library page behave like folder navigation with explicit book-processing states. Adding a book should create a visible pending entry first, then the user advances it through Initialize Book, Build Registry, Annotate Book, and finally Open when it is TTS-ready.

## State Model

Each web-added book gets `readalong_book.json` at its root. The manifest stores title, slug, original EPUB filename, source EPUB path, completed stages, and last-read location. Pending books are discovered by this manifest even before `chapters/` exists.

States:

- `fresh_added`: source EPUB saved, no extracted chapters. Action: `Initialize Book`.
- `initialized`: chapters, toc, sentence segments, and registry shell exist. Action: `Build Registry`.
- `registry_ready`: global registry pass completed. Action: `Annotate Book`.
- `annotated`: all chapters have quote attribution annotations. Action: `Open`.

Existing books without a manifest are summarized from artifacts for backward compatibility.

## Actions

Add Book saves `_source/original.epub`, writes the manifest, and refreshes the library row. It does not open the reader or extract chapters.

Initialize Book calls the existing EPUB extraction/segmentation path.

Build Registry calls the existing global registry pass and marks the registry stage complete.

Annotate Book runs annotation for all chapters and builds read-along units.

Open is available only when all chapters are annotated. This prevents missing annotation warnings in the reader.

## Last Read

The manifest stores the most recent read-along chapter and unit id. Session start records the selected starting unit, and session advance records the consumed unit.
