from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple, Union
from zipfile import ZipFile

from ebook_tts_pipeline.paths import BookPaths


CHAPTER_MEMBER_RE = re.compile(r"(chapter|prologue|epilogue|part)[-_]?[0-9ivxlcdm]*", re.IGNORECASE)
CHAPTER_TEXT_RE = re.compile(r"(?im)^\s*(chapter\s+\S+|prologue|epilogue|part\s+\S+)\s*$")


@dataclass(frozen=True)
class EpubExtractResult:
    chapters: List[str]
    sources: List[str]


class _HtmlToText(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


class EpubChapterExtractor:
    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars

    def extract(self, epub_path: Union[str, Path], paths: BookPaths) -> EpubExtractResult:
        epub = Path(epub_path)
        chapters: List[str] = []
        sources: List[str] = []
        with ZipFile(epub) as zf:
            opf_path = self._find_opf_path(zf)
            docs = self._spine_documents(zf, opf_path)
            filename_chapters = [doc for doc in docs if self._member_looks_like_chapter(doc[0])]
            text_chapters = [doc for doc in docs if self._text_looks_like_chapter(doc[1])]
            selected = filename_chapters or text_chapters or [
                (member, text) for member, text in docs if len(text) >= self.min_chars
            ]

            for index, (member, text) in enumerate(selected, start=1):
                if len(text) < self.min_chars:
                    continue
                chapter_id = f"chapter_{index:03d}"
                output = paths.chapter_text(chapter_id)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(text.rstrip() + "\n", encoding="utf-8")
                chapters.append(chapter_id)
                sources.append(member)

        return EpubExtractResult(chapters=chapters, sources=sources)

    def _find_opf_path(self, zf: ZipFile) -> str:
        if "META-INF/container.xml" in zf.namelist():
            root = ET.fromstring(zf.read("META-INF/container.xml"))
            for elem in root.iter():
                if _local_name(elem.tag) == "rootfile":
                    full_path = elem.attrib.get("full-path")
                    if full_path:
                        return full_path
        if "content.opf" in zf.namelist():
            return "content.opf"
        raise ValueError("EPUB package file (.opf) not found")

    def _spine_documents(self, zf: ZipFile, opf_path: str) -> List[Tuple[str, str]]:
        opf_root = ET.fromstring(zf.read(opf_path))
        manifest: Dict[str, str] = {}
        for elem in opf_root.iter():
            if _local_name(elem.tag) == "item":
                media_type = elem.attrib.get("media-type", "")
                if "html" in media_type:
                    item_id = elem.attrib.get("id")
                    href = elem.attrib.get("href")
                    if item_id and href:
                        manifest[item_id] = href

        opf_dir = posixpath.dirname(opf_path)
        docs: List[Tuple[str, str]] = []
        for elem in opf_root.iter():
            if _local_name(elem.tag) != "itemref":
                continue
            href = manifest.get(elem.attrib.get("idref", ""))
            if not href:
                continue
            member = posixpath.normpath(posixpath.join(opf_dir, href))
            if member not in zf.namelist():
                continue
            html = zf.read(member).decode("utf-8", errors="replace")
            parser = _HtmlToText()
            parser.feed(html)
            docs.append((member, parser.text()))
        return docs

    def _member_looks_like_chapter(self, member: str) -> bool:
        basename = posixpath.basename(member)
        return bool(CHAPTER_MEMBER_RE.search(basename))

    def _text_looks_like_chapter(self, text: str) -> bool:
        return bool(CHAPTER_TEXT_RE.search(text[:500]))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
