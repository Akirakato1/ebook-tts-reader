from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.global_registry import GlobalRegistryService
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.epub_ingestion import EpubChapterExtractor, EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.registry import build_compact_voice_profile, prune_deprecated_registry_fields
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter
from ebook_tts_pipeline.voice_identity import append_differentiators


class ChapterStage(str, Enum):
    RAW = "raw"
    SEGMENTED = "segmented"
    ANNOTATED = "annotated"
    SCRIPTED = "scripted"
    AUDIO = "audio"


@dataclass(frozen=True)
class ChapterRow:
    chapter: str
    index: int
    title: str
    stage: ChapterStage
    audio_path: Path


@dataclass(frozen=True)
class BookLibraryEntry:
    title: str
    slug: str
    book_root: Path
    epub_path: Path


@dataclass(frozen=True)
class RegistryField:
    key: str
    label: str
    value: str
    multiline: bool = False


@dataclass(frozen=True)
class RegistryCharacterForm:
    role_id: str
    title: str
    readonly_fields: List[RegistryField]
    editable_fields: List[RegistryField]


@dataclass(frozen=True)
class ChapterActionResult:
    chapter: str
    stage: ChapterStage
    message: str


class ChapterExtractor(Protocol):
    def extract(self, epub_path: Union[str, Path], paths: BookPaths) -> EpubExtractResult:
        ...


PipelineFactory = Callable[[PipelineConfig, bool, bool], AudiobookPipeline]
AudioOpener = Callable[[Path], None]


class PrototypeUiController:
    def __init__(
        self,
        book_root: Union[str, Path],
        pipeline_factory: Optional[PipelineFactory] = None,
        extractor: Optional[ChapterExtractor] = None,
        audio_opener: Optional[AudioOpener] = None,
        fake_tts: bool = False,
        library_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.book_root = Path(book_root)
        self.paths = BookPaths(self.book_root)
        self.library_path = Path(library_path) if library_path else Path("books") / "library.json"
        self.current_book_slug = ""
        self.pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.extractor = extractor or EpubChapterExtractor()
        self.audio_opener = audio_opener or _open_audio_file
        self.fake_tts = fake_tts
        self._sync_current_book_from_library()

    def set_book_root(self, book_root: Union[str, Path]) -> None:
        self.book_root = Path(book_root)
        self.paths = BookPaths(self.book_root)
        self._sync_current_book_from_library()

    def library_books(self) -> List[BookLibraryEntry]:
        if not self.library_path.exists():
            return []
        payload = read_json(self.library_path)
        books = []
        for item in payload.get("books", []):
            book_root = str(item.get("book_root", "")).strip()
            if not book_root:
                continue
            books.append(
                BookLibraryEntry(
                    title=str(item.get("title", "")).strip() or str(item.get("slug", "")).strip() or book_root,
                    slug=str(item.get("slug", "")).strip() or Path(book_root).name,
                    book_root=Path(book_root),
                    epub_path=Path(str(item.get("epub_path", "")).strip()),
                )
            )
        return books

    def select_book(self, slug: str) -> BookLibraryEntry:
        for book in self.library_books():
            if book.slug == slug:
                self.current_book_slug = book.slug
                self.set_book_root(book.book_root)
                return book
        raise ValueError(f"Book not found in library: {slug}")

    def chapter_rows(self) -> List[ChapterRow]:
        chapters_dir = self.book_root / "chapters"
        toc = self._load_toc()
        if not chapters_dir.exists():
            return []
        rows: List[ChapterRow] = []
        for index, chapter_file in enumerate(sorted(chapters_dir.glob("*.txt")), start=1):
            chapter = chapter_file.stem
            rows.append(
                ChapterRow(
                    chapter=chapter,
                    index=index,
                    title=toc.get(chapter, self._chapter_title(chapter_file, chapter)),
                    stage=self.chapter_stage(chapter),
                    audio_path=self.paths.chapter_audio(chapter),
                )
            )
        return rows

    def chapter_stage(self, chapter: str) -> ChapterStage:
        if self.paths.chapter_audio(chapter).exists():
            return ChapterStage.AUDIO
        if self.paths.tts_script(chapter).exists() and self.paths.qwen_script(chapter).exists():
            return ChapterStage.SCRIPTED
        if self.paths.annotation(chapter).exists():
            return ChapterStage.ANNOTATED
        if self.paths.sentence_artifact(chapter).exists():
            return ChapterStage.SEGMENTED
        return ChapterStage.RAW

    def registry_text(self) -> str:
        if not self.paths.registry.exists():
            return "{}\n"
        return json.dumps(read_json(self.paths.registry), indent=2, ensure_ascii=False) + "\n"

    def save_registry_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Registry JSON is invalid: {exc}") from exc
        prune_deprecated_registry_fields(payload)
        write_json_atomic(self.paths.registry, payload)

    def registry_character_forms(self) -> List[RegistryCharacterForm]:
        if not self.paths.registry.exists():
            return []
        registry = read_json(self.paths.registry)
        forms = []
        for role_id, character in sorted(registry.get("characters", {}).items()):
            identity = dict(character.get("identity_profile", character.get("character_profile", {})))
            voice_identity = dict(character.get("voice_identity", {}))
            variants = dict(character.get("voice_variants", {}))
            qvp_paths = [
                str(variant.get("voice_config_path", ""))
                for variant in variants.values()
                if variant.get("voice_config_path")
            ]
            forms.append(
                RegistryCharacterForm(
                    role_id=role_id,
                    title=str(character.get("display_name", role_id)),
                    readonly_fields=[
                        RegistryField("role_id", "Role ID", str(character.get("role_id", role_id))),
                        RegistryField("profile_id", "Profile ID", str(character.get("profile_id", ""))),
                        RegistryField("person_id", "Person ID", str(character.get("person_id", ""))),
                        RegistryField("seed", "Seed", str(voice_identity.get("seed", ""))),
                        RegistryField("voice_config_path", "Voice Files", ", ".join(qvp_paths)),
                    ],
                    editable_fields=[
                        RegistryField("display_name", "Character Name", str(character.get("display_name", ""))),
                        RegistryField("age", "Age", _field_text(character.get("age", identity.get("age")))),
                        RegistryField("age_stage", "Age Stage", str(character.get("age_stage", identity.get("age_stage", "")))),
                        RegistryField("gender", "Gender", str(identity.get("gender", ""))),
                        RegistryField("personality", "Personality", ", ".join(_string_list(identity.get("personality")))),
                        RegistryField("race_or_ethnicity", "Race / Ethnicity", _field_text(identity.get("race_or_ethnicity"))),
                        RegistryField("accent", "Accent", _field_text(identity.get("accent"))),
                        RegistryField("occupation", "Occupation", _field_text(identity.get("occupation"))),
                        RegistryField("aliases", "Aliases", ", ".join(_string_list(character.get("aliases")))),
                    ],
                )
            )
        return forms

    def save_registry_character_form(self, role_id: str, values: Dict[str, str]) -> None:
        registry = read_json(self.paths.registry)
        characters = registry.get("characters", {})
        if role_id not in characters:
            raise ValueError(f"Registry character not found: {role_id}")

        character = characters[role_id]
        identity = dict(character.get("identity_profile", character.get("character_profile", {})))
        display_name = values.get("display_name", str(character.get("display_name", role_id))).strip() or role_id
        age = _parse_age(values.get("age", ""))
        age_stage = values.get("age_stage", str(identity.get("age_stage", "unknown"))).strip() or "unknown"
        gender = values.get("gender", str(identity.get("gender", "unknown"))).strip() or "unknown"
        personality = _split_csv(values.get("personality", ""))

        identity.update(
            {
                "age": age,
                "age_stage": age_stage,
                "gender": gender,
                "personality": personality,
                "race_or_ethnicity": _blank_to_none(values.get("race_or_ethnicity", "")),
                "accent": _blank_to_none(values.get("accent", "")),
                "occupation": _blank_to_none(values.get("occupation", "")),
            }
        )
        character["display_name"] = display_name
        character["age"] = age
        character["age_stage"] = age_stage
        character["aliases"] = _split_csv(values.get("aliases", ""))
        character["identity_profile"] = identity
        self._refresh_character_voice_profiles(character)
        prune_deprecated_registry_fields(registry)
        write_json_atomic(self.paths.registry, registry)

    def load_epub(self, epub_path: Union[str, Path], title: str, slug: str) -> EpubExtractResult:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        result = self.extractor.extract(epub_path, self.paths)
        pipeline = self._pipeline(needs_llm=False)
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug)
        self.current_book_slug = slug
        toc = []
        for index, chapter in enumerate(result.chapters, start=1):
            pipeline.segment_chapter(chapter)
            source = result.sources[index - 1] if index - 1 < len(result.sources) else ""
            toc.append(
                {
                    "index": index,
                    "chapter": chapter,
                    "title": self._chapter_title(self.paths.chapter_text(chapter), chapter),
                    "source": source,
                }
            )
        write_json_atomic(self.book_root / "toc.json", {"chapters": toc})
        self._register_library_book(
            BookLibraryEntry(
                title=title,
                slug=slug,
                book_root=self.book_root,
                epub_path=Path(epub_path),
            )
        )
        return result

    def run_next_chapter_action(self, chapter: str) -> ChapterActionResult:
        stage = self.chapter_stage(chapter)
        if stage == ChapterStage.AUDIO:
            self.audio_opener(self.paths.chapter_audio(chapter))
            return ChapterActionResult(chapter=chapter, stage=stage, message="Opened audio file.")

        if stage == ChapterStage.SCRIPTED:
            pipeline = self._pipeline(needs_llm=False)
            annotation = self._load_annotation(chapter)
            pipeline.prepare_voices_for_annotation(annotation)
            pipeline.synthesize_chapter_from_tts_script(chapter)
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.AUDIO, message="Generated audio.")

        if stage == ChapterStage.ANNOTATED:
            pipeline = self._pipeline(needs_llm=False)
            pipeline.build_sentence_jobs(chapter, self._load_annotation(chapter))
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.SCRIPTED, message="Generated scripts.")

        pipeline = self._pipeline(needs_llm=True)
        if stage == ChapterStage.RAW:
            pipeline.segment_chapter(chapter)
        pipeline.annotate_chapter(chapter, lock_registry=True)
        return ChapterActionResult(chapter=chapter, stage=ChapterStage.ANNOTATED, message="Annotated chapter.")

    def build_global_registry(self) -> int:
        pipeline = self._pipeline(needs_llm=True)
        registry = pipeline.registry.load()
        count = pipeline.build_global_registry(
            book_title=str(registry.get("book", {}).get("title", "Untitled Book")),
        )
        return count

    def _pipeline(self, needs_llm: bool) -> AudiobookPipeline:
        return self.pipeline_factory(PipelineConfig.from_env(str(self.book_root)), needs_llm, self.fake_tts)

    def _load_annotation(self, chapter: str) -> AnnotationResult:
        return AnnotationResult.from_dict(read_json(self.paths.annotation(chapter)))

    def _load_toc(self) -> Dict[str, str]:
        toc_path = self.book_root / "toc.json"
        if not toc_path.exists():
            return {}
        payload = read_json(toc_path)
        return {
            str(item.get("chapter")): str(item.get("title"))
            for item in payload.get("chapters", [])
            if item.get("chapter") and item.get("title")
        }

    def _chapter_title(self, chapter_file: Path, fallback: str) -> str:
        if not chapter_file.exists():
            return fallback
        for line in chapter_file.read_text(encoding="utf-8", errors="replace").splitlines():
            title = " ".join(line.split())
            if title:
                return title[:120]
        return fallback

    def _register_library_book(self, entry: BookLibraryEntry) -> None:
        existing = [book for book in self.library_books() if book.slug != entry.slug]
        existing.append(entry)
        existing.sort(key=lambda book: book.title.lower())
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.library_path,
            {
                "books": [
                    {
                        "title": book.title,
                        "slug": book.slug,
                        "book_root": str(book.book_root),
                        "epub_path": str(book.epub_path),
                    }
                    for book in existing
                ]
            },
        )

    def _sync_current_book_from_library(self) -> None:
        resolved_root = self.book_root.resolve()
        for book in self.library_books():
            try:
                if book.book_root.resolve() == resolved_root:
                    self.current_book_slug = book.slug
                    return
            except OSError:
                continue
        self.current_book_slug = ""

    def _refresh_character_voice_profiles(self, character: Dict[str, Any]) -> None:
        identity = dict(character.get("identity_profile", {}))
        display_name = str(character.get("display_name", character.get("role_id", "Character")))
        voice = build_compact_voice_profile(display_name, {"identity_profile": identity})
        differentiators = list(character.get("voice_identity", {}).get("differentiators", []))
        if differentiators:
            voice["qwen_instruct"] = append_differentiators(str(voice["qwen_instruct"]), differentiators)

        variants = character.setdefault("voice_variants", {})
        default = variants.get("default")
        if isinstance(default, dict):
            default["display_name"] = f"{display_name}_default"
            default["voice_profile"] = voice
            default["voice_config_hash"] = None
        internal = variants.get("internal")
        if isinstance(internal, dict):
            internal["display_name"] = f"{display_name}_internal"
            internal["voice_profile"] = _internal_voice_profile(display_name, voice)
            internal["voice_config_hash"] = None


def _default_pipeline_factory(config: PipelineConfig, needs_llm: bool, fake_tts: bool) -> AudiobookPipeline:
    return AudiobookPipeline(
        config=config,
        annotation_service=AnnotationService(
            client=_build_llm_client(config) if needs_llm else _UnavailableJsonClient(),
            repair_retries=config.annotation_repair_retries,
            failure_logger=FailureLogger(
                config.debug_log_root,
                context={"book_root": config.book_root},
            ),
        ),
        global_registry_service=GlobalRegistryService(
            client=_build_llm_client(config) if needs_llm else _UnavailableJsonClient(),
            failure_logger=FailureLogger(
                config.debug_log_root,
                context={"book_root": config.book_root},
            ),
        ),
        tts_adapter=FakeTtsAdapter() if fake_tts else _build_qwen_adapter(config),
    )


def _build_llm_client(config: PipelineConfig) -> AnthropicJsonClient:
    return AnthropicJsonClient(
        api_key=config.require_anthropic_key(),
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )


def _build_qwen_adapter(config: PipelineConfig) -> QwenTtsAdapter:
    return QwenTtsAdapter(
        model_root=config.qwen_model_root,
        model_choice=config.qwen_model_choice,
        device=config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
        max_batch_size=config.qwen_batch_size,
    )


def _open_audio_file(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


class _UnavailableJsonClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        raise RuntimeError("This UI action does not perform LLM annotation.")


def _internal_voice_profile(display_name: str, base_voice: Dict[str, Any]) -> Dict[str, str]:
    description = str(base_voice.get("description", "")).rstrip(". ")
    qwen_instruct = str(base_voice.get("qwen_instruct", "")).rstrip(". ")
    return {
        "description": (
            f"{description}; same {display_name} identity for internal monologue, "
            "closer, softer, reflective, less projected"
        ).strip("; "),
        "qwen_instruct": (
            f"{qwen_instruct}. Keep the same {display_name} speaker identity and timbre, "
            "but perform this as internal monologue: closer, softer, inward, reflective, "
            "and less projected. Do not whisper unless the text itself implies whispering."
        ).strip(),
    }


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _blank_to_none(value: str) -> Optional[str]:
    text = str(value).strip()
    return text or None


def _parse_age(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return None
    return int(text) if text.isdigit() else text


def _split_csv(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
