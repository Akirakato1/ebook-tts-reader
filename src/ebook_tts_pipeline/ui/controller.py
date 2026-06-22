from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.global_registry import GlobalRegistryService
from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionService
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.epub_ingestion import EpubChapterExtractor, EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    migrate_registry_voice_records,
    prune_deprecated_registry_fields,
)
from ebook_tts_pipeline.registry import normalize_name
from ebook_tts_pipeline.temp_registry import normalize_annotation_local_speakers
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter
from ebook_tts_pipeline.voice_identity import append_differentiators


class ChapterStage(str, Enum):
    RAW = "raw"
    SEGMENTED = "segmented"
    ANNOTATION_REVIEW = "annotation_review"
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
class AgeStageOption:
    age_stage: str
    role_name: str
    role_id: str


@dataclass(frozen=True)
class AnnotationAppearanceForm:
    key: str
    name: str
    current_age_stage: str
    current_role_name: str
    age_stage_options: List[AgeStageOption]


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
            if not self._annotation_is_approved(chapter):
                return ChapterStage.ANNOTATION_REVIEW
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
        migrate_registry_voice_records(payload)
        prune_deprecated_registry_fields(payload)
        write_json_atomic(self.paths.registry, payload)

    def tts_settings(self) -> Dict[str, Any]:
        config = PipelineConfig.from_env(str(self.book_root))
        defaults = {
            "qwen_batch_size": config.qwen_batch_size,
            "tts_speed": config.tts_speed,
            "pause_between_sentences_ms": config.pause_between_sentences_ms,
            "intra_sentence_pause_ms": config.intra_sentence_pause_ms,
        }
        if not self.paths.settings.exists():
            return defaults
        payload = read_json(self.paths.settings)
        return {
            "qwen_batch_size": _positive_int(payload.get("qwen_batch_size"), defaults["qwen_batch_size"]),
            "tts_speed": _positive_float(payload.get("tts_speed"), defaults["tts_speed"]),
            "pause_between_sentences_ms": _nonnegative_int(
                payload.get("pause_between_sentences_ms"),
                defaults["pause_between_sentences_ms"],
            ),
            "intra_sentence_pause_ms": _nonnegative_int(
                payload.get("intra_sentence_pause_ms"),
                defaults["intra_sentence_pause_ms"],
            ),
        }

    def save_tts_settings(self, values: Dict[str, Any]) -> None:
        settings = {
            "qwen_batch_size": _positive_int(values.get("qwen_batch_size"), 24),
            "tts_speed": _positive_float(values.get("tts_speed"), 1.0),
            "pause_between_sentences_ms": _nonnegative_int(values.get("pause_between_sentences_ms"), 250),
            "intra_sentence_pause_ms": _nonnegative_int(values.get("intra_sentence_pause_ms"), 50),
        }
        write_json_atomic(self.paths.settings, settings)

    def registry_character_forms(self) -> List[RegistryCharacterForm]:
        if not self.paths.registry.exists():
            return []
        registry = read_json(self.paths.registry)
        migrate_registry_voice_records(registry)
        forms = []
        for role_id, character in sorted(registry.get("characters", {}).items()):
            identity = dict(character.get("identity_profile", character.get("character_profile", {})))
            voice_identity = dict(character.get("voice_identity", {}))
            qvp_path = str(character.get("voice_config_path", "") or "")
            forms.append(
                RegistryCharacterForm(
                    role_id=role_id,
                    title=str(character.get("display_name", role_id)),
                    readonly_fields=[
                        RegistryField("role_id", "Role ID", str(character.get("role_id", role_id))),
                        RegistryField("profile_id", "Profile ID", str(character.get("profile_id", ""))),
                        RegistryField("person_id", "Person ID", str(character.get("person_id", ""))),
                        RegistryField("seed", "Seed", str(voice_identity.get("seed", ""))),
                        RegistryField("voice_config_path", "Voice File", qvp_path),
                    ],
                    editable_fields=[
                        RegistryField("display_name", "Character Name", str(character.get("display_name", ""))),
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
        age_stage = values.get("age_stage", str(identity.get("age_stage", "unknown"))).strip() or "unknown"
        gender = values.get("gender", str(identity.get("gender", "unknown"))).strip() or "unknown"
        personality = _split_csv(values.get("personality", ""))

        identity.update(
            {
                "age_stage": age_stage,
                "gender": gender,
                "personality": personality,
                "race_or_ethnicity": _blank_to_none(values.get("race_or_ethnicity", "")),
                "accent": _blank_to_none(values.get("accent", "")),
                "occupation": _blank_to_none(values.get("occupation", "")),
            }
        )
        character["display_name"] = display_name
        character["age_stage"] = age_stage
        character["aliases"] = _split_csv(values.get("aliases", ""))
        character["identity_profile"] = identity
        self._refresh_character_voice_profiles(character)
        migrate_registry_voice_records(registry)
        prune_deprecated_registry_fields(registry)
        write_json_atomic(self.paths.registry, registry)

    def annotation_appearance_forms(self, chapter: str) -> List[AnnotationAppearanceForm]:
        if not self.paths.annotation(chapter).exists() or not self.paths.registry.exists():
            return []
        annotation = self._load_annotation(chapter)
        registry = read_json(self.paths.registry)
        forms: List[AnnotationAppearanceForm] = []
        seen_person_ids = set()

        for role_name in annotation.roles:
            if normalize_name(role_name) == normalize_name("Narrator"):
                continue
            match = _find_character_record_by_role_name(registry, role_name)
            if match is None:
                continue
            _, record = match
            person_id = str(record.get("person_id", "")).strip() or str(record.get("role_id", role_name))
            if person_id in seen_person_ids:
                continue
            options = _age_stage_options_for_person(registry, person_id)
            if not options:
                continue
            seen_person_ids.add(person_id)
            forms.append(
                AnnotationAppearanceForm(
                    key=person_id,
                    name=str(record.get("display_name", role_name)),
                    current_age_stage=_character_age_stage(record),
                    current_role_name=_annotation_role_name_for_record(record),
                    age_stage_options=options,
                )
            )

        return forms

    def confirm_annotation_appearances(self, chapter: str, selections: Dict[str, str]) -> None:
        if not self.paths.annotation(chapter).exists():
            raise ValueError(f"Annotation does not exist for {chapter}.")
        raw_annotation = read_json(self.paths.annotation(chapter))
        annotation = AnnotationResult.from_dict(raw_annotation)
        annotation = self._normalize_annotation_local_speakers(chapter, annotation)
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        forms = self.annotation_appearance_forms(chapter)
        option_by_person: Dict[str, Dict[str, AgeStageOption]] = {
            form.key: {option.age_stage: option for option in form.age_stage_options}
            for form in forms
        }
        selected_options: Dict[str, AgeStageOption] = {}

        for form in forms:
            selected_age_stage = selections.get(form.key, form.current_age_stage)
            options = option_by_person[form.key]
            if selected_age_stage not in options:
                raise ValueError(
                    f"Invalid age stage for {form.name}: {selected_age_stage}. "
                    f"Choose one of {', '.join(options)}."
                )
            selected_options[form.key] = options[selected_age_stage]

        old_roles = list(annotation.roles)
        new_roles: List[str] = []
        old_to_new_role_idx: Dict[int, int] = {}
        for role_idx, role_name in enumerate(old_roles):
            replacement = role_name
            if normalize_name(role_name) != normalize_name("Narrator"):
                match = _find_character_record_by_role_name(registry, role_name)
                if match is not None:
                    _, record = match
                    person_id = str(record.get("person_id", "")).strip()
                    if person_id in selected_options:
                        replacement = selected_options[person_id].role_name
            if replacement not in new_roles:
                new_roles.append(replacement)
            old_to_new_role_idx[role_idx] = new_roles.index(replacement)

        rewritten = AnnotationResult(
            new_characters=annotation.new_characters,
            roles=new_roles,
            types=annotation.types,
            script=[
                (old_to_new_role_idx[role_idx], type_idx, sentence_idx)
                for role_idx, type_idx, sentence_idx in annotation.script
            ],
            local_speakers=annotation.local_speakers,
            proposed_new_characters=annotation.proposed_new_characters,
        )
        payload = rewritten.to_dict()
        if _is_quote_annotation_payload(raw_annotation):
            payload["schema"] = str(raw_annotation.get("schema", "quote_attribution_v1"))
            payload["quotes"] = [
                [int(quote_idx), old_to_new_role_idx[int(role_idx)], str(quote_type)]
                for quote_idx, role_idx, quote_type in raw_annotation.get("quotes", [])
            ]
        write_json_atomic(self.paths.annotation(chapter), payload)
        write_json_atomic(
            self.paths.annotation_approval(chapter),
            {
                "chapter": chapter,
                "approved": True,
                "appearances": [
                    {
                        "person_id": form.key,
                        "name": form.name,
                        "age_stage": selected_options[form.key].age_stage,
                        "role_name": selected_options[form.key].role_name,
                    }
                    for form in forms
                ],
            },
        )
        if new_roles != old_roles:
            self._invalidate_downstream_artifacts(chapter)

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
            annotation = self._normalize_annotation_local_speakers(chapter, self._load_annotation(chapter))
            pipeline.prepare_voices_for_annotation(annotation, chapter=chapter)
            pipeline.synthesize_chapter_from_tts_script(chapter)
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.AUDIO, message="Generated audio.")

        if stage == ChapterStage.ANNOTATION_REVIEW:
            return ChapterActionResult(
                chapter=chapter,
                stage=stage,
                message="Review and confirm character appearances before generating scripts.",
            )

        if stage == ChapterStage.ANNOTATED:
            pipeline = self._pipeline(needs_llm=False)
            annotation = self._normalize_annotation_local_speakers(chapter, self._load_annotation(chapter))
            pipeline.build_sentence_jobs(chapter, annotation)
            return ChapterActionResult(chapter=chapter, stage=ChapterStage.SCRIPTED, message="Generated scripts.")

        pipeline = self._pipeline(needs_llm=True)
        if stage == ChapterStage.RAW:
            pipeline.segment_chapter(chapter)
        pipeline.annotate_chapter(chapter, lock_registry=True)
        return ChapterActionResult(
            chapter=chapter,
            stage=ChapterStage.ANNOTATION_REVIEW,
            message="Annotated chapter. Review character appearances before scripts.",
        )

    def build_global_registry(self) -> int:
        pipeline = self._pipeline(needs_llm=True)
        title, slug = self._registry_book_metadata()
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug)
        count = pipeline.build_global_registry(book_title=title)
        return count

    def _pipeline(self, needs_llm: bool) -> AudiobookPipeline:
        settings = self.tts_settings()
        config = replace(
            PipelineConfig.from_env(str(self.book_root)),
            qwen_batch_size=settings["qwen_batch_size"],
            tts_speed=settings["tts_speed"],
            pause_between_sentences_ms=settings["pause_between_sentences_ms"],
            intra_sentence_pause_ms=settings["intra_sentence_pause_ms"],
        )
        return self.pipeline_factory(config, needs_llm, self.fake_tts)

    def _load_annotation(self, chapter: str) -> AnnotationResult:
        return AnnotationResult.from_dict(read_json(self.paths.annotation(chapter)))

    def _annotation_is_approved(self, chapter: str) -> bool:
        approval_path = self.paths.annotation_approval(chapter)
        if not approval_path.exists():
            return False
        try:
            return bool(read_json(approval_path).get("approved"))
        except Exception:
            return False

    def _normalize_annotation_local_speakers(
        self,
        chapter: str,
        annotation: AnnotationResult,
    ) -> AnnotationResult:
        normalized = normalize_annotation_local_speakers(annotation)
        if normalized != annotation:
            write_json_atomic(self.paths.annotation(chapter), normalized.to_dict())
        return normalized

    def _invalidate_downstream_artifacts(self, chapter: str) -> None:
        for path in [
            self.paths.tts_script(chapter),
            self.paths.qwen_script(chapter),
            self.paths.chapter_audio(chapter),
            self.paths.chapter_timeline(chapter),
        ]:
            if path.exists():
                path.unlink()

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

    def _registry_book_metadata(self) -> Tuple[str, str]:
        if self.paths.registry.exists():
            registry = read_json(self.paths.registry)
            book = dict(registry.get("book", {}))
            title = str(book.get("title", "")).strip()
            slug = str(book.get("slug", "")).strip()
            if title and slug:
                return title, slug

        resolved_root = self.book_root.resolve()
        for book in self.library_books():
            try:
                root_matches = book.book_root.resolve() == resolved_root
            except OSError:
                root_matches = False
            if root_matches or (self.current_book_slug and book.slug == self.current_book_slug):
                return book.title or "Untitled Book", book.slug or self.book_root.name or "book"

        return self.book_root.name or "Untitled Book", self.current_book_slug or self.book_root.name or "book"

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
        character["voice_profile"] = voice
        character["voice_config_hash"] = None
        character.pop("voice_variants", None)


def _find_character_record_by_role_name(registry: Dict[str, Any], role_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    characters = {
        str(role_id): record
        for role_id, record in registry.get("characters", {}).items()
        if isinstance(record, dict)
    }
    normalized = normalize_name(role_name)
    display_counts: Dict[str, int] = {}
    for record in characters.values():
        display_name = str(record.get("display_name", ""))
        if not display_name:
            continue
        normalized_display = normalize_name(display_name)
        display_counts[normalized_display] = display_counts.get(normalized_display, 0) + 1

    for role_id, record in characters.items():
        names = _character_exact_role_names(record)
        if normalized in {normalize_name(name) for name in names if name}:
            return role_id, record

    for role_id, record in characters.items():
        display_name = str(record.get("display_name", ""))
        if display_name and display_counts[normalize_name(display_name)] == 1 and normalized == normalize_name(display_name):
            return role_id, record

    return None


def _character_exact_role_names(record: Dict[str, Any]) -> List[str]:
    names = [
        str(record.get("role_id", "")),
        str(record.get("role_id", "")).replace("_", " "),
        str(record.get("profile_id", "")),
        str(record.get("profile_id", "")).replace("_", " "),
    ]
    names.extend(str(alias) for alias in record.get("aliases", []))
    return names


def _age_stage_options_for_person(registry: Dict[str, Any], person_id: str) -> List[AgeStageOption]:
    options: List[AgeStageOption] = []
    for role_id, record in registry.get("characters", {}).items():
        if not isinstance(record, dict):
            continue
        if str(record.get("person_id", "")).strip() != person_id:
            continue
        options.append(
            AgeStageOption(
                age_stage=_character_age_stage(record),
                role_name=_annotation_role_name_for_record(record),
                role_id=str(record.get("role_id", role_id)),
            )
        )
    return options


def _annotation_role_name_for_record(record: Dict[str, Any]) -> str:
    display_name = str(record.get("display_name", "")).strip()
    age_stage = _character_age_stage(record)
    preferred = f"{display_name} {age_stage.replace('_', ' ')}".strip()
    for alias in record.get("aliases", []):
        if normalize_name(str(alias)) == normalize_name(preferred):
            return str(alias)
    if display_name and age_stage and age_stage != "unknown":
        return preferred
    role_id = str(record.get("role_id", "")).strip()
    return role_id.replace("_", " ") if role_id else display_name


def _character_age_stage(record: Dict[str, Any]) -> str:
    identity = dict(record.get("identity_profile", {})) if isinstance(record.get("identity_profile"), dict) else {}
    return str(record.get("age_stage") or identity.get("age_stage") or "unknown").strip().lower().replace(" ", "_")


def _is_quote_annotation_payload(payload: Dict[str, Any]) -> bool:
    return payload.get("schema") == "quote_attribution_v1" or "quotes" in payload


def _default_pipeline_factory(config: PipelineConfig, needs_llm: bool, fake_tts: bool) -> AudiobookPipeline:
    quote_client = _build_llm_client(config) if needs_llm else _UnavailableJsonClient()
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
        quote_attribution_service=QuoteAttributionService(quote_client) if needs_llm else None,
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


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _blank_to_none(value: str) -> Optional[str]:
    text = str(value).strip()
    return text or None


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
