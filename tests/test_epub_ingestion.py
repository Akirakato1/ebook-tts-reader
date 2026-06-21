from zipfile import ZIP_DEFLATED, ZipFile

from ebook_tts_pipeline.epub_ingestion import EpubChapterExtractor
from ebook_tts_pipeline.paths import BookPaths


def test_epub_extractor_writes_chapter_named_spine_documents(tmp_path):
    epub_path = tmp_path / "sample.epub"
    with ZipFile(epub_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
              <manifest>
                <item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>
                <item id="c1" href="text/chapter001.xhtml" media-type="application/xhtml+xml"/>
                <item id="c2" href="text/chapter002.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="title"/>
                <itemref idref="c1"/>
                <itemref idref="c2"/>
              </spine>
            </package>
            """,
        )
        zf.writestr("title.xhtml", "<html><body><h1>Title Page</h1></body></html>")
        zf.writestr(
            "text/chapter001.xhtml",
            "<html><body><h1>Chapter 1</h1><p>The first room was silent.</p></body></html>",
        )
        zf.writestr(
            "text/chapter002.xhtml",
            "<html><body><h1>Chapter 2</h1><p>The second room was loud.</p></body></html>",
        )

    paths = BookPaths(tmp_path / "book")
    result = EpubChapterExtractor().extract(epub_path, paths)

    assert result.chapters == ["chapter_001", "chapter_002"]
    assert paths.chapter_text("chapter_001").read_text(encoding="utf-8").startswith("Chapter 1")
    assert "second room" in paths.chapter_text("chapter_002").read_text(encoding="utf-8")


def test_epub_extractor_does_not_treat_contents_page_as_chapter_when_filenames_match(tmp_path):
    epub_path = tmp_path / "sample.epub"
    with ZipFile(epub_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
              <manifest>
                <item id="toc" href="contents.xhtml" media-type="application/xhtml+xml"/>
                <item id="c1" href="chapter001.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="toc"/>
                <itemref idref="c1"/>
              </spine>
            </package>
            """,
        )
        zf.writestr(
            "contents.xhtml",
            "<html><body><h1>Contents</h1><p>Chapter 1</p><p>Chapter 2</p></body></html>",
        )
        zf.writestr(
            "chapter001.xhtml",
            "<html><body><h1>Chapter 1</h1><p>The actual opening chapter.</p></body></html>",
        )

    paths = BookPaths(tmp_path / "book")
    result = EpubChapterExtractor().extract(epub_path, paths)

    assert result.chapters == ["chapter_001"]
    assert result.sources == ["chapter001.xhtml"]
    assert "actual opening" in paths.chapter_text("chapter_001").read_text(encoding="utf-8")
