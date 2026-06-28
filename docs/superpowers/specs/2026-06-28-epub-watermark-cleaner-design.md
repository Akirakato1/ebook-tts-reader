# EPUB Watermark Cleaner Design

## Goal

Add a Python CLI command that removes OceanofPDF watermark artifacts from an EPUB and writes a cleaned EPUB file. The cleaner should preserve book formatting, fonts, assets, metadata, filenames, and archive layout as much as practical.

## Scope

The command targets the watermark pattern seen in the sample EPUB:

- a standalone archive member named `oceanofpdf.com`
- visible HTML/XHTML watermark blocks or links containing `OceanofPDF.com` and `https://oceanofpdf.com`

The cleaner does not attempt general piracy-site cleanup, text rewriting, typographic normalization, or EPUB repair beyond preserving a valid output archive.

## CLI

Expose the feature as a subcommand of the existing console script:

```powershell
ebook-tts clean-epub input.epub --output output.cleaned.epub
```

If `--output` is omitted, the output path defaults to `<input-stem>.cleaned.epub` next to the input file. The command refuses to overwrite the input path.

## Architecture

Create a focused EPUB cleaning module near the existing EPUB ingestion code. The module reads the input EPUB as a zip archive and writes a new archive entry by entry.

For every member:

1. Skip the standalone `oceanofpdf.com` member.
2. For HTML-like text members, decode as UTF-8, remove only known OceanofPDF watermark snippets, then re-encode as UTF-8.
3. Copy all other members byte-for-byte.

The writer preserves each copied member's filename, timestamp, comment, extra data, external attributes, and compression settings. The EPUB `mimetype` entry stays uncompressed.

## Error Handling

The command reports a clear error for missing inputs, non-EPUB paths, unreadable zip files, and output paths that equal the input path. If an HTML-like member cannot be decoded, it is copied unchanged rather than risking damage.

## Tests

Tests cover:

- removal of the standalone `oceanofpdf.com` member
- removal of a trailing XHTML watermark block
- preservation of unrelated archive members byte-for-byte
- default output naming
- refusal to overwrite the input EPUB
