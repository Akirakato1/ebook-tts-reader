# Readalongweb EPUB Upload Design

## Goal

Make adding a book in the read-along web UI feel like a normal desktop app: the user clicks Browse, selects an EPUB, confirms title/folder, and the app creates and initializes the book.

## Approach

Browsers do not expose a reliable local filesystem path from a file picker, so the visible UI uses an upload flow instead of asking for a path. The server is local, but the browser still sends the selected EPUB bytes through `multipart/form-data`.

The existing JSON path-based add-book endpoint stays available for tests and power users. The normal browser UI sends multipart data to the same `/api/library/add-book` endpoint.

## Flow

1. User clicks Browse and selects an `.epub`.
2. The UI auto-fills title and folder slug from the filename if those fields are blank.
3. User clicks Add Book.
4. Server creates `<library_root>/<slug>/_source/original.epub`.
5. Server calls the existing `PrototypeUiController.load_epub()` with that saved EPUB.
6. Server selects the new book and returns the same active-book payload used by the existing add flow.

## Error Handling

The server rejects multipart requests without an EPUB file, blank title/slug values are filled from the filename, and existing non-empty target folders are rejected before writing a book. Uploads are written only inside the selected library root.

## Tests

Tests verify that the HTML uses a file input, that multipart upload creates a book folder and `_source/original.epub`, and that a multipart request without an EPUB returns a JSON error.
