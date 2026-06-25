from __future__ import annotations

import json
import time
import webbrowser
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Union

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.global_registry import GlobalRegistryService
from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionService
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig, resolve_project_path, resolve_qwen_model_root
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.domain import AnnotationResult
from ebook_tts_pipeline.epub_ingestion import EpubChapterExtractor, EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.read_along.narrator_profile import (
    functional_narrator_voice_record,
    narrator_profile_from_registry,
    narrator_profile_hash,
    narrator_summary,
    narrator_voice_record,
    normalize_narrator_profile,
)
from ebook_tts_pipeline.read_along.session import ReadAlongSession
from ebook_tts_pipeline.read_along.units import (
    FUNCTIONAL_NARRATOR_ROLE,
    FUNCTIONAL_NARRATOR_ROLE_ID,
    FUNCTIONAL_NARRATOR_VARIANT,
    ReadAlongUnit,
)
from ebook_tts_pipeline.registry import (
    build_compact_voice_profile,
    migrate_registry_voice_records,
    prune_deprecated_registry_fields,
    voice_profile_hash,
)
from ebook_tts_pipeline.registry import normalize_name
from ebook_tts_pipeline.runtime_logging import log_runtime_step
from ebook_tts_pipeline.temp_registry import ChapterTempRegistryManager, normalize_annotation_local_speakers
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio, TtsAdapter
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter
from ebook_tts_pipeline.tts.vllm_omni_adapter import WslVllmOmniQwenAdapter
from ebook_tts_pipeline.tts.wsl_adapter import WslQwenWorkerAdapter
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
    author: str = ""


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


class _LazyTtsAdapter:
    def __init__(self, adapter_factory: Callable[[], TtsAdapter]) -> None:
        self._adapter_factory = adapter_factory
        self._adapter: Optional[TtsAdapter] = None

    def _require_adapter(self) -> TtsAdapter:
        if self._adapter is None:
            self._adapter = self._adapter_factory()
        return self._adapter

    def ensure_voice(self, role_id: str, voice_record: Dict, voice_path: Path) -> Path:
        return self._require_adapter().ensure_voice(role_id, voice_record, voice_path)

    def generate_sentence_batches(self, jobs: List[Dict]) -> Iterator[List[GeneratedSentenceAudio]]:
        yield from self._require_adapter().generate_sentence_batches(jobs)

    def generate_sentences(self, jobs: List[Dict]) -> List[GeneratedSentenceAudio]:
        return self._require_adapter().generate_sentences(jobs)

    def close(self) -> None:
        if self._adapter is None:
            return
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()


ACCENT_OPTIONS = [
    "",
    "General American",
    "Southern American",
    "British",
    "Irish",
    "Scottish",
    "Australian",
    "Canadian",
    "New York",
    "Tokyo",
    "Custom",
]

RACE_OR_ETHNICITY_OPTIONS = [
    "",
    "African American",
    "British",
    "Chinese",
    "Hispanic / Latino",
    "Indian",
    "Irish",
    "Japanese",
    "Korean",
    "White",
    "Custom",
]


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
                    author=str(item.get("author", "")).strip(),
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
            "tts_speed": config.tts_speed,
            "pause_between_sentences_ms": config.pause_between_sentences_ms,
            "intra_sentence_pause_ms": config.intra_sentence_pause_ms,
        }
        if not self.paths.settings.exists():
            return defaults
        payload = read_json(self.paths.settings)
        return {
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
            "tts_speed": _positive_float(values.get("tts_speed"), 1.0),
            "pause_between_sentences_ms": _nonnegative_int(values.get("pause_between_sentences_ms"), 250),
            "intra_sentence_pause_ms": _nonnegative_int(values.get("intra_sentence_pause_ms"), 50),
        }
        write_json_atomic(self.paths.settings, settings)

    def read_along_narrator_profile(self) -> Dict[str, Any]:
        if self.paths.read_along_narrator_profile.exists():
            return normalize_narrator_profile(read_json(self.paths.read_along_narrator_profile))
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        book_slug = str(registry.get("book", {}).get("slug", self.book_root.name))
        profile = narrator_profile_from_registry(registry, book_slug=book_slug)
        write_json_atomic(self.paths.read_along_narrator_profile, profile)
        return profile

    def save_read_along_narrator_profile(self, values: Dict[str, Any]) -> Dict[str, Any]:
        current = self.read_along_narrator_profile()
        identity = dict(current.get("identity_profile") or {})
        identity.update(
            {
                "age_stage": str(values.get("age_stage", identity.get("age_stage", "adult"))).strip() or "adult",
                "gender": str(values.get("gender", identity.get("gender", "unknown"))).strip() or "unknown",
                "personality": _split_csv(values.get("personality", ",".join(identity.get("personality", [])))),
                "race_or_ethnicity": _blank_to_none(
                    values.get("race_or_ethnicity", identity.get("race_or_ethnicity", ""))
                ),
                "accent": _blank_to_none(values.get("accent", identity.get("accent", ""))),
                "occupation": _blank_to_none(values.get("occupation", identity.get("occupation", "audiobook narrator"))),
            }
        )
        profile = normalize_narrator_profile(
            {
                "role_id": "narrator",
                "display_name": str(values.get("display_name", current.get("display_name", "Narrator"))).strip()
                or "Narrator",
                "identity_profile": identity,
                "voice_identity": dict(current.get("voice_identity") or {}),
            },
            book_slug=self.book_root.name,
        )
        write_json_atomic(self.paths.read_along_narrator_profile, profile)
        return profile

    def read_along_narrator_profile_payload(self) -> Dict[str, Any]:
        profile = self.read_along_narrator_profile()
        identity = dict(profile.get("identity_profile") or {})
        return {
            "profile": profile,
            "summary": narrator_summary(profile),
            "hash": narrator_profile_hash(profile),
            "fields": {
                "display_name": str(profile.get("display_name", "Narrator")),
                "age_stage": str(identity.get("age_stage", "")),
                "gender": str(identity.get("gender", "")),
                "personality": ", ".join(identity.get("personality", [])),
                "race_or_ethnicity": str(identity.get("race_or_ethnicity") or ""),
                "accent": str(identity.get("accent") or ""),
                "occupation": str(identity.get("occupation") or ""),
            },
        }

    def read_along_settings(self) -> Dict[str, Any]:
        defaults = {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20.0,
            "start_buffer_seconds": 20.0,
            "max_buffer_seconds": 40.0,
            "max_buffer_units": 32,
        }
        if not self.paths.read_along_settings.exists():
            return defaults
        payload = read_json(self.paths.read_along_settings)
        return {
            "playback_speed": _bounded_positive_float(
                payload.get("playback_speed"),
                defaults["playback_speed"],
                1.0,
                4.0,
            ),
            "generation_mode": _choice(
                payload.get("generation_mode"),
                {"precise", "balanced", "fast"},
                defaults["generation_mode"],
            ),
            "buffer_limit": min(
                8,
                max(1, _positive_int(payload.get("buffer_limit"), defaults["buffer_limit"])),
            ),
            "target_buffer_seconds": _bounded_positive_float(
                payload.get("target_buffer_seconds"),
                defaults["target_buffer_seconds"],
                0.1,
                120.0,
            ),
            "start_buffer_seconds": _bounded_positive_float(
                payload.get("start_buffer_seconds"),
                defaults["start_buffer_seconds"],
                0.1,
                120.0,
            ),
            "max_buffer_seconds": _bounded_positive_float(
                payload.get("max_buffer_seconds"),
                defaults["max_buffer_seconds"],
                0.1,
                240.0,
            ),
            "max_buffer_units": min(
                32,
                max(1, _positive_int(payload.get("max_buffer_units"), defaults["max_buffer_units"])),
            ),
        }

    def save_read_along_settings(self, values: Dict[str, Any]) -> None:
        target_buffer_seconds = _bounded_positive_float(values.get("target_buffer_seconds"), 20.0, 0.1, 120.0)
        start_buffer_seconds = _bounded_positive_float(values.get("start_buffer_seconds"), 20.0, 0.1, 120.0)
        max_buffer_seconds = _bounded_positive_float(values.get("max_buffer_seconds"), 40.0, 0.1, 240.0)
        if start_buffer_seconds > target_buffer_seconds:
            start_buffer_seconds = target_buffer_seconds
        if target_buffer_seconds > max_buffer_seconds:
            max_buffer_seconds = target_buffer_seconds
        settings = {
            "playback_speed": _bounded_positive_float(values.get("playback_speed"), 1.0, 1.0, 4.0),
            "generation_mode": _choice(values.get("generation_mode"), {"precise", "balanced", "fast"}, "balanced"),
            "buffer_limit": min(8, max(1, _positive_int(values.get("buffer_limit"), 2))),
            "target_buffer_seconds": target_buffer_seconds,
            "start_buffer_seconds": start_buffer_seconds,
            "max_buffer_seconds": max_buffer_seconds,
            "max_buffer_units": min(32, max(1, _positive_int(values.get("max_buffer_units"), 32))),
        }
        write_json_atomic(self.paths.read_along_settings, settings)

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

    def registry_review_payload(self) -> Dict[str, Any]:
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        entries: List[Dict[str, Any]] = []
        for form in self.registry_character_forms():
            fields = {field.key: field.value for field in form.editable_fields}
            voice_path = ""
            for field in form.readonly_fields:
                if field.key == "voice_config_path":
                    voice_path = field.value
                    break
            entries.append(
                {
                    "kind": "character",
                    "role_id": form.role_id,
                    "title": form.title,
                    "editable": True,
                    "fields": fields,
                    "voice_config_path": voice_path,
                    "sample_url": self._voice_sample_url(form.role_id),
                }
            )
        return {
            "book": dict(registry.get("book", {})),
            "accent_options": _merged_detected_options(
                ACCENT_OPTIONS,
                _detected_identity_values(self.paths, registry, "accent"),
            ),
            "race_or_ethnicity_options": _merged_detected_options(
                RACE_OR_ETHNICITY_OPTIONS,
                _detected_identity_values(self.paths, registry, "race_or_ethnicity"),
            ),
            "entries": entries,
        }

    def _voice_sample_url(self, role_id: str) -> str:
        sample_path = self.paths.root / "voices" / "_samples" / f"{role_id}.wav"
        return f"/api/registry/sample/{role_id}.wav" if sample_path.exists() else ""

    def generate_registry_voice_sample(
        self,
        role_id: str,
        pipeline: Optional[AudiobookPipeline] = None,
    ) -> Dict[str, str]:
        role_id = str(role_id).strip()
        if not role_id:
            raise ValueError("role_id is required")
        registry = read_json(self.paths.registry)
        record = self._registry_voice_record(role_id, registry)
        display_name = str(record.get("display_name", role_id)).strip() or role_id
        owns_pipeline = pipeline is None
        if pipeline is None:
            pipeline = self._voice_asset_pipeline()
        if hasattr(pipeline, "_voice_path_for_record"):
            voice_path = pipeline._voice_path_for_record(role_id, record)
        else:
            voice_path = self.paths.voice_qvp(role_id)
        try:
            pipeline.tts_adapter.ensure_voice(role_id, record, voice_path)
            record["voice_config_path"] = voice_path.relative_to(self.paths.root).as_posix()
            record["voice_config_hash"] = voice_profile_hash(record)
            pipeline.registry.save(registry)

            generated = pipeline.tts_adapter.generate_sentences(
                [
                    {
                        "index": 0,
                        "sentence_idx": 0,
                        "unit_idx": 0,
                        "text": f"Hi, my name is {display_name}.",
                        "role": display_name,
                        "role_id": role_id,
                        "type": "dialogue",
                        "voice_config_path": record["voice_config_path"],
                    }
                ]
            )
            if not generated:
                raise RuntimeError("Voice sample generation returned no audio.")
            sample_dir = self.paths.root / "voices" / "_samples"
            sample_path = sample_dir / f"{role_id}.wav"
            _write_wav_file(sample_path, generated[0].samples, generated[0].sample_rate)
            return {
                "role_id": role_id,
                "sample_path": sample_path.relative_to(self.paths.root).as_posix(),
                "sample_url": f"/api/registry/sample/{role_id}.wav",
            }
        finally:
            if owns_pipeline:
                close = getattr(pipeline.tts_adapter, "close", None)
                if callable(close):
                    close()

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
        is_quote_annotation = _is_quote_annotation_payload(raw_annotation)
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
        if is_quote_annotation:
            payload = dict(raw_annotation)
            payload["schema"] = str(raw_annotation.get("schema", "quote_attribution_v1"))
            payload["roles"] = new_roles
            payload["quotes"] = [
                _compact_quote_row_for_annotation(row, old_to_new_role_idx)
                for row in raw_annotation.get("quotes", [])
            ]
            if rewritten.local_speakers:
                payload["local_speakers"] = rewritten.local_speakers
            else:
                payload.pop("local_speakers", None)
            payload.pop("types", None)
            payload.pop("script", None)
        else:
            payload = rewritten.to_dict()
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

    def load_epub(
        self,
        epub_path: Union[str, Path],
        title: str,
        slug: str,
        author: str = "",
    ) -> EpubExtractResult:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        result = self.extractor.extract(epub_path, self.paths)
        pipeline = self._pipeline(needs_llm=False, read_along=True)
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug, book_author=author)
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
                author=author,
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

    def build_read_along_units(self, chapter: str) -> List[Dict[str, Any]]:
        pipeline = self._pipeline(needs_llm=False)
        return pipeline.build_read_along_units(chapter)

    def read_along_units(self, chapter: str) -> List[Dict[str, Any]]:
        if not self.paths.read_along_units(chapter).exists():
            return self.build_read_along_units(chapter)
        return list(read_json(self.paths.read_along_units(chapter)).get("units", []))

    def chapter_text(self, chapter: str) -> str:
        return self.paths.chapter_text(chapter).read_text(encoding="utf-8", errors="replace")

    def create_read_along_session(
        self,
        chapter: str,
        units: List[Dict[str, Any]],
        settings: Dict[str, Any],
        store_audio_files: bool = True,
    ) -> ReadAlongSession:
        session_id = f"{chapter}-{int(time.time())}"
        pipeline = self._pipeline(needs_llm=False, read_along=True)
        units = pipeline.build_read_along_units(chapter)
        session_voice_paths = self._ensure_read_along_session_narrator_voices(
            pipeline,
            settings,
            session_id,
            units,
        )
        units = self._apply_session_narrator_voice_paths(units, session_voice_paths)
        units = self._ensure_read_along_session_temp_voices(pipeline, chapter, units)
        write_json_atomic(self.paths.read_along_units(chapter), {"chapter": chapter, "units": units})
        missing = self._missing_read_along_voice_paths([chapter])
        if missing:
            raise ValueError("Prepare Voices before starting a read-along session.")
        self._register_session_narrator_voice_paths(pipeline.tts_adapter, session_voice_paths)
        return ReadAlongSession(
            session_id=session_id,
            units=[ReadAlongUnit.from_dict(unit) for unit in units],
            tts_adapter=pipeline.tts_adapter,
            session_dir=self.paths.read_along_session_dir(session_id),
            timing_log_path=self.paths.read_along_timing_log(session_id),
            buffer_limit=int(settings["buffer_limit"]),
            playback_speed=float(settings["playback_speed"]),
            generation_mode=str(settings["generation_mode"]),
            target_buffer_seconds=float(settings.get("target_buffer_seconds", 20.0)),
            start_buffer_seconds=float(settings.get("start_buffer_seconds", 20.0)),
            max_buffer_seconds=float(settings.get("max_buffer_seconds", 40.0)),
            max_buffer_units=int(settings.get("max_buffer_units", 32)),
            store_audio_files=store_audio_files,
        )

    def process_read_along_book(self) -> Dict[str, int]:
        pipeline = self._pipeline(needs_llm=True)
        title, slug = self._registry_book_metadata()
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug)
        registry_count = pipeline.build_global_registry(book_title=title)
        chapters = [row.chapter for row in self.chapter_rows()]
        segmented = 0
        annotated = 0
        units_built = 0
        for chapter in chapters:
            if not self.paths.sentence_artifact(chapter).exists():
                pipeline.segment_chapter(chapter)
                segmented += 1
            annotation_payload = read_json(self.paths.annotation(chapter)) if self.paths.annotation(chapter).exists() else {}
            if not _is_quote_annotation_payload(annotation_payload):
                pipeline.annotate_chapter(chapter, lock_registry=True)
                annotated += 1
            pipeline.build_read_along_units(chapter)
            units_built += 1
        return {
            "registry_characters": registry_count,
            "chapters": len(chapters),
            "segmented": segmented,
            "annotated": annotated,
            "units_built": units_built,
        }

    def annotate_read_along_book(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, int]:
        pipeline = self._pipeline(needs_llm=True)
        chapters = [row.chapter for row in self.chapter_rows()]
        start_index = self._annotation_resume_start_index(chapters)
        annotated = 0
        units_built = 0
        self._write_annotation_progress(
            {
                "status": "running",
                "completed": start_index,
                "total": len(chapters),
                "current_chapter": "",
                "failed_chapter": "",
                "error": "",
            }
        )
        for index, chapter in enumerate(chapters[start_index:], start=start_index + 1):
            started_event = {
                "chapter": chapter,
                "index": index,
                "total": len(chapters),
                "status": "started",
            }
            self._emit_annotation_progress(started_event, progress_callback)
            self._write_annotation_progress(
                {
                    "status": "running",
                    "completed": index - 1,
                    "total": len(chapters),
                    "current_chapter": chapter,
                    "failed_chapter": "",
                    "error": "",
                }
            )
            try:
                if not self.paths.sentence_artifact(chapter).exists():
                    pipeline.segment_chapter(chapter)
                annotation_payload = read_json(self.paths.annotation(chapter)) if self.paths.annotation(chapter).exists() else {}
                if not _is_quote_annotation_payload(annotation_payload):
                    pipeline.annotate_chapter(chapter, lock_registry=True)
                    annotated += 1
                pipeline.build_read_along_units(chapter)
                units_built += 1
            except Exception as exc:
                error = str(exc)
                failed_event = {
                    "chapter": chapter,
                    "index": index,
                    "total": len(chapters),
                    "status": "failed",
                    "error": error,
                }
                self._emit_annotation_progress(failed_event, progress_callback)
                self._write_annotation_progress(
                    {
                        "status": "failed",
                        "completed": index - 1,
                        "total": len(chapters),
                        "current_chapter": chapter,
                        "failed_chapter": chapter,
                        "error": error,
                    }
                )
                raise RuntimeError(f"Annotation failed at {chapter}: {error}") from exc
            completed_event = {"chapter": chapter, "index": index, "total": len(chapters), "status": "completed"}
            self._emit_annotation_progress(completed_event, progress_callback)
            self._write_annotation_progress(
                {
                    "status": "running",
                    "completed": index,
                    "total": len(chapters),
                    "current_chapter": chapter,
                    "failed_chapter": "",
                    "error": "",
                }
            )
        self._write_annotation_progress(
            {
                "status": "completed",
                "completed": len(chapters),
                "total": len(chapters),
                "current_chapter": "",
                "failed_chapter": "",
                "error": "",
            }
        )
        return {
            "chapters": len(chapters),
            "annotated": annotated,
            "units_built": units_built,
        }

    def _annotation_resume_start_index(self, chapters: List[str]) -> int:
        progress_path = self.paths.root / "read_along" / "annotation_progress.json"
        progress: Dict[str, Any] = {}
        if progress_path.exists():
            try:
                payload = read_json(progress_path)
                progress = payload if isinstance(payload, dict) else {}
            except Exception:
                progress = {}

        status = str(progress.get("status") or "")
        if status in {"failed", "running"}:
            for key in ("failed_chapter", "current_chapter"):
                chapter = str(progress.get(key) or "")
                if chapter in chapters:
                    return chapters.index(chapter)
            try:
                completed = int(progress.get("completed") or 0)
            except (TypeError, ValueError):
                completed = 0
            if 0 < completed <= len(chapters):
                return completed - 1

        for index, chapter in enumerate(chapters):
            annotation_payload = read_json(self.paths.annotation(chapter)) if self.paths.annotation(chapter).exists() else {}
            if not _is_quote_annotation_payload(annotation_payload):
                return index
            if not self.paths.read_along_units(chapter).exists():
                return index
        return 0

    def _emit_annotation_progress(
        self,
        event: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if progress_callback is not None:
            progress_callback(dict(event))

    def _write_annotation_progress(self, payload: Dict[str, Any]) -> None:
        progress_path = self.paths.root / "read_along" / "annotation_progress.json"
        write_json_atomic(progress_path, payload)

    def prepare_read_along_voices(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        pipeline = self._voice_asset_pipeline()
        chapters = [row.chapter for row in self.chapter_rows()]
        voice_count, voice_total = self._registry_voice_readiness_counts()
        self._emit_voice_progress(
            {
                "phase": "chapters",
                "status": "started",
                "completed": voice_count,
                "total": voice_total,
            },
            progress_callback,
        )
        prepared_chapters = 0
        try:
            for chapter in chapters:
                if not self.paths.annotation(chapter).exists():
                    raise ValueError(f"Annotation missing for {chapter}.")
            sample_count = self.generate_registry_voice_samples(
                pipeline=pipeline,
                progress_callback=progress_callback,
            )
            voice_count, voice_total = self._registry_voice_readiness_counts()
            for chapter in chapters:
                self._emit_voice_progress(
                    {
                        "phase": "chapters",
                        "status": "started",
                        "chapter": chapter,
                        "completed": voice_count,
                        "total": voice_total,
                    },
                    progress_callback,
                )
                pipeline.build_read_along_units(chapter)
                prepared_chapters += 1
            voice_count, voice_total = self._registry_voice_readiness_counts()
            missing = self._missing_read_along_voice_paths(chapters)
            return {
                "chapters": len(chapters),
                "prepared_chapters": prepared_chapters,
                "sample_count": sample_count,
                "voice_count": voice_count,
                "voice_total": voice_total,
                "missing_voice_paths": missing,
                "voices_ready": not missing and voice_count >= voice_total,
            }
        finally:
            close = getattr(pipeline.tts_adapter, "close", None)
            if callable(close):
                close()

    def generate_registry_voice_samples(
        self,
        pipeline: Optional[AudiobookPipeline] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        role_ids = self._registry_character_role_ids(registry)
        generated = 0
        for role_id in role_ids:
            record = dict(registry.get("characters", {}).get(role_id, {}))
            self.generate_registry_voice_sample(role_id, pipeline=pipeline)
            generated += 1
            log_runtime_step(
                "readalong_voice_sample_ready",
                role_id=role_id,
                completed=generated,
                total=len(role_ids),
            )
            self._emit_voice_progress(
                {
                    "phase": "samples",
                    "status": "completed",
                    "role_id": role_id,
                    "display_name": record.get("display_name", role_id),
                    "completed": generated,
                    "total": len(role_ids),
                },
                progress_callback,
            )
        return generated

    def _emit_voice_progress(
        self,
        event: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if progress_callback is not None:
            progress_callback(dict(event))

    def _ensure_read_along_session_narrator_voices(
        self,
        pipeline: AudiobookPipeline,
        _settings: Dict[str, Any],
        _session_id: str,
        units: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        profile = self.read_along_narrator_profile()
        narrator_record = narrator_voice_record(profile)
        profile_hash = voice_profile_hash(narrator_record)
        narrator_record["voice_config_hash"] = profile_hash
        narrator_path = self.paths.narrator_voice_qvp(profile_hash, "narrator")
        narrator_rel = self._ensure_voice_asset(
            pipeline.tts_adapter,
            "narrator",
            narrator_record,
            narrator_path,
        )

        voice_paths = {"narrator": narrator_rel}
        if any(_is_functional_narrator_unit(unit) for unit in units):
            functional_record = functional_narrator_voice_record(profile)
            functional_hash = voice_profile_hash(functional_record)
            functional_record["voice_config_hash"] = functional_hash
            functional_path = self.paths.narrator_voice_qvp(functional_hash, FUNCTIONAL_NARRATOR_ROLE_ID)
            voice_paths[FUNCTIONAL_NARRATOR_ROLE_ID] = self._ensure_voice_asset(
                pipeline.tts_adapter,
                FUNCTIONAL_NARRATOR_ROLE_ID,
                functional_record,
                functional_path,
            )
        return voice_paths

    def _ensure_voice_asset(
        self,
        adapter: Any,
        role_id: str,
        record: Dict[str, Any],
        voice_path: Path,
    ) -> str:
        current_hash = voice_profile_hash(record)
        cached_hash = record.get("voice_config_hash")
        should_generate = not voice_path.exists() or cached_hash != current_hash
        if should_generate:
            adapter_record = dict(record)
            adapter_record["_force_regenerate"] = voice_path.exists() and cached_hash != current_hash
            adapter.ensure_voice(role_id, adapter_record, voice_path)
            record["voice_config_hash"] = current_hash
        if hasattr(adapter, "role_voice_paths"):
            adapter.role_voice_paths[role_id] = voice_path
        record["voice_config_path"] = voice_path.relative_to(self.paths.root).as_posix()
        return str(record["voice_config_path"])

    def _apply_session_narrator_voice_paths(
        self,
        units: List[Dict[str, Any]],
        voice_paths: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        patched: List[Dict[str, Any]] = []
        for unit in units:
            item = dict(unit)
            if _is_functional_narrator_unit(item):
                item["role"] = FUNCTIONAL_NARRATOR_ROLE
                item["role_id"] = FUNCTIONAL_NARRATOR_ROLE_ID
                item["type"] = "narration"
                item["voice_variant"] = FUNCTIONAL_NARRATOR_VARIANT
                item["voice_config_path"] = voice_paths[FUNCTIONAL_NARRATOR_ROLE_ID]
            elif str(item.get("role_id") or "") == "narrator" or str(item.get("role") or "") == "Narrator":
                item["voice_config_path"] = voice_paths["narrator"]
            patched.append(item)
        return patched

    def _ensure_read_along_session_temp_voices(
        self,
        pipeline: AudiobookPipeline,
        chapter: str,
        units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        temp_units = [dict(unit) for unit in units if _is_temp_voice_unit(unit)]
        if not temp_units:
            return units
        temp_manager = ChapterTempRegistryManager(self.paths)
        temp_registry = temp_manager.load(chapter)
        speakers = temp_registry.get("speakers", {})
        if not isinstance(speakers, dict):
            return units

        patched_paths: Dict[str, str] = {}
        for unit in temp_units:
            current_path = str(unit.get("voice_config_path") or "").strip()
            speaker = _temp_speaker_for_unit(speakers, unit, current_path)
            if speaker is None:
                continue
            role_id = str(speaker.get("role_id") or unit.get("role_id") or "").strip()
            if not role_id:
                continue
            voice_path = self.paths.root / current_path
            log_runtime_step(
                "readalong_session_temp_voice",
                chapter=chapter,
                role_id=role_id,
                voice_path=voice_path,
            )
            patched_paths[role_id] = self._ensure_voice_asset(
                pipeline.tts_adapter,
                role_id,
                speaker,
                voice_path,
            )
        temp_manager.save(chapter, temp_registry)
        if not patched_paths:
            return units

        patched: List[Dict[str, Any]] = []
        for unit in units:
            item = dict(unit)
            role_id = str(item.get("role_id") or "")
            if role_id in patched_paths:
                item["voice_config_path"] = patched_paths[role_id]
            patched.append(item)
        return patched

    def _register_session_narrator_voice_paths(
        self,
        adapter: Any,
        voice_paths: Dict[str, str],
    ) -> None:
        role_voice_paths = getattr(adapter, "role_voice_paths", None)
        if not isinstance(role_voice_paths, dict):
            return
        narrator_path = self.paths.root / voice_paths["narrator"]
        role_voice_paths["Narrator"] = narrator_path
        role_voice_paths["narrator"] = narrator_path
        if FUNCTIONAL_NARRATOR_ROLE_ID in voice_paths:
            functional_path = self.paths.root / voice_paths[FUNCTIONAL_NARRATOR_ROLE_ID]
            role_voice_paths[FUNCTIONAL_NARRATOR_ROLE] = functional_path
            role_voice_paths[FUNCTIONAL_NARRATOR_ROLE_ID] = functional_path

    def build_global_registry(self) -> int:
        pipeline = self._pipeline(needs_llm=True)
        title, slug = self._registry_book_metadata()
        pipeline.registry.initialize_if_missing(book_title=title, book_slug=slug)
        count = pipeline.build_global_registry(book_title=title)
        return count

    def _pipeline(self, needs_llm: bool, read_along: bool = False) -> AudiobookPipeline:
        settings = self.tts_settings()
        base_config = PipelineConfig.from_env(str(self.book_root))
        config = replace(
            base_config,
            tts_speed=settings["tts_speed"],
            pause_between_sentences_ms=settings["pause_between_sentences_ms"],
            intra_sentence_pause_ms=settings["intra_sentence_pause_ms"],
        )
        if read_along:
            config = replace(config, tts_backend=base_config.read_along_tts_backend)
        return self.pipeline_factory(config, needs_llm, self.fake_tts)

    def _voice_asset_pipeline(self) -> AudiobookPipeline:
        settings = self.tts_settings()
        base_config = PipelineConfig.from_env(str(self.book_root))
        backend = base_config.voice_asset_tts_backend
        if backend in {"wsl-vllm-omni", "vllm-omni"}:
            backend = "wsl"
        log_runtime_step(
            "voice_asset_pipeline",
            backend=backend,
            book_root=self.book_root,
            qwen_model_root=resolve_qwen_model_root(base_config.qwen_model_root),
        )
        config = replace(
            base_config,
            tts_backend=backend,
            tts_speed=settings["tts_speed"],
            pause_between_sentences_ms=settings["pause_between_sentences_ms"],
            intra_sentence_pause_ms=settings["intra_sentence_pause_ms"],
        )
        return self.pipeline_factory(config, False, self.fake_tts)

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
                        "author": book.author,
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

    def _registry_voice_record(self, role_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        if role_id == "narrator":
            return registry.setdefault("narrator", {"role_id": "narrator", "display_name": "Narrator"})
        characters = registry.setdefault("characters", {})
        if role_id not in characters:
            raise ValueError(f"Registry character not found: {role_id}")
        return characters[role_id]

    def _registry_voice_readiness_counts(self) -> Tuple[int, int]:
        registry = read_json(self.paths.registry) if self.paths.registry.exists() else {}
        role_ids = self._registry_character_role_ids(registry)
        ready = 0
        for role_id in role_ids:
            record = registry.get("characters", {}).get(role_id, {})
            voice_path = str(record.get("voice_config_path") or "").strip()
            sample_path = self.paths.root / "voices" / "_samples" / f"{role_id}.wav"
            if voice_path and (self.paths.root / voice_path).exists() and sample_path.exists():
                ready += 1
        return ready, len(role_ids)

    def _registry_character_role_ids(self, registry: Dict[str, Any]) -> List[str]:
        return [
            str(role_id)
            for role_id, record in registry.get("characters", {}).items()
            if isinstance(record, dict)
        ]

    def _missing_read_along_voice_paths(self, chapters: List[str]) -> List[str]:
        missing: List[str] = []
        for chapter in chapters:
            units_path = self.paths.read_along_units(chapter)
            if not units_path.exists():
                missing.append(f"{chapter}:read_along_units")
                continue
            for unit in read_json(units_path).get("units", []):
                if _is_session_narrator_unit(unit) or _is_temp_voice_unit(unit):
                    continue
                voice_path = str(unit.get("voice_config_path") or "").strip()
                unit_id = unit.get("unit_id")
                if not voice_path:
                    missing.append(f"{chapter}:unit_{unit_id}:empty")
                    continue
                if not (self.paths.root / voice_path).exists():
                    missing.append(f"{chapter}:unit_{unit_id}:{voice_path}")
        return missing


def _is_functional_narrator_unit(unit: Dict[str, Any]) -> bool:
    return (
        str(unit.get("role_id") or "") == FUNCTIONAL_NARRATOR_ROLE_ID
        or str(unit.get("voice_variant") or "") == FUNCTIONAL_NARRATOR_VARIANT
    )


def _is_session_narrator_unit(unit: Dict[str, Any]) -> bool:
    return _is_functional_narrator_unit(unit) or str(unit.get("role_id") or "") == "narrator"


def _is_temp_voice_unit(unit: Dict[str, Any]) -> bool:
    return str(unit.get("voice_config_path") or "").strip().replace("\\", "/").startswith("voices/_temp/")


def _temp_speaker_for_unit(
    speakers: Dict[str, Any],
    unit: Dict[str, Any],
    voice_config_path: str,
) -> Optional[Dict[str, Any]]:
    role_id = str(unit.get("role_id") or "")
    normalized_path = voice_config_path.replace("\\", "/")
    for speaker in speakers.values():
        if not isinstance(speaker, dict):
            continue
        if str(speaker.get("role_id") or "") == role_id:
            return speaker
        if str(speaker.get("voice_config_path") or "").replace("\\", "/") == normalized_path:
            return speaker
    return None


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


def _compact_quote_row_for_annotation(row: Any, old_to_new_role_idx: Dict[int, int]) -> List[Any]:
    values = list(row)
    quote_idx = int(values[0])
    role_idx = old_to_new_role_idx[int(values[1])]
    quote_type = str(values[2]) if len(values) > 2 else "dialogue"
    if quote_type == "dialogue":
        return [quote_idx, role_idx]
    return [quote_idx, role_idx, quote_type]


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
        quote_attribution_service=(
            QuoteAttributionService(
                quote_client,
                failure_logger=FailureLogger(
                    config.debug_log_root,
                    context={"book_root": config.book_root},
                ),
            )
            if needs_llm
            else None
        ),
        tts_adapter=FakeTtsAdapter() if fake_tts else _LazyTtsAdapter(lambda: _build_qwen_adapter(config)),
    )


def _build_llm_client(config: PipelineConfig) -> AnthropicJsonClient:
    return AnthropicJsonClient(
        api_key=config.require_anthropic_key(),
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )


def _build_qwen_adapter(config: PipelineConfig) -> QwenTtsAdapter:
    qwen_model_root = resolve_qwen_model_root(config.qwen_model_root)
    log_runtime_step(
        "build_tts_adapter",
        backend=config.tts_backend,
        book_root=config.book_root,
        qwen_model_root=qwen_model_root,
    )
    if config.tts_backend in {"wsl-vllm-omni", "vllm-omni"}:
        stage_configs_path = resolve_project_path(config.vllm_omni_stage_configs_path)
        log_runtime_step(
            "build_tts_adapter_vllm_omni",
            model=_resolve_vllm_omni_model(config),
            stage_config=stage_configs_path,
            voice_model_root=qwen_model_root,
            wsl_distro=config.wsl_distro,
            wsl_python=config.vllm_omni_wsl_python,
        )
        return WslVllmOmniQwenAdapter(
            book_root=config.book_root,
            distro=config.wsl_distro,
            python_path=config.vllm_omni_wsl_python,
            model=_resolve_vllm_omni_model(config),
            stage_configs_path=stage_configs_path,
            voice_model_root=qwen_model_root,
            voice_python_path=config.wsl_python,
            voice_model_choice=config.qwen_model_choice,
            voice_device="cuda" if config.qwen_device == "auto" else config.qwen_device,
            voice_precision=config.qwen_precision,
            voice_attention=config.qwen_attention,
            timeout_seconds=config.wsl_timeout_seconds,
        )
    if config.tts_backend == "wsl":
        log_runtime_step(
            "build_tts_adapter_wsl_qwen",
            model_root=qwen_model_root,
            wsl_distro=config.wsl_distro,
            wsl_python=config.wsl_python,
            attention=config.qwen_attention,
            precision=config.qwen_precision,
        )
        return WslQwenWorkerAdapter(
            book_root=config.book_root,
            model_root=qwen_model_root,
            distro=config.wsl_distro,
            python_path=config.wsl_python,
            model_choice=config.qwen_model_choice,
            device="cuda" if config.qwen_device == "auto" else config.qwen_device,
            precision=config.qwen_precision,
            attention=config.qwen_attention,
            max_new_tokens=config.qwen_max_new_tokens,
            max_generation_block_chars=config.qwen_max_generation_block_chars,
            max_generation_blocks_per_call=config.qwen_max_generation_blocks_per_call,
            cache_clear_interval=config.qwen_cache_clear_interval,
            streaming_text_mode=config.qwen_streaming_text_mode,
            performance_log_path=Path(config.qwen_perf_log_path) if config.qwen_perf_log_path else None,
            adaptive_memory_target_bytes=(
                int(config.qwen_adaptive_memory_target_gb * (1024 ** 3))
                if config.qwen_adaptive_memory_target_gb is not None
                else None
            ),
            timeout_seconds=config.wsl_timeout_seconds,
        )
    return QwenTtsAdapter(
        model_root=str(qwen_model_root),
        model_choice=config.qwen_model_choice,
        device=config.qwen_device,
        precision=config.qwen_precision,
        attention=config.qwen_attention,
        max_new_tokens=config.qwen_max_new_tokens,
        max_generation_block_chars=config.qwen_max_generation_block_chars,
        max_generation_blocks_per_call=config.qwen_max_generation_blocks_per_call,
        cache_clear_interval=config.qwen_cache_clear_interval,
        streaming_text_mode=config.qwen_streaming_text_mode,
        performance_log_path=config.qwen_perf_log_path,
        adaptive_memory_target_bytes=(
            int(config.qwen_adaptive_memory_target_gb * (1024 ** 3))
            if config.qwen_adaptive_memory_target_gb is not None
            else None
        ),
    )


def _resolve_vllm_omni_model(config: PipelineConfig) -> Path | str:
    local_model = resolve_qwen_model_root(config.qwen_model_root) / "Qwen3-TTS-12Hz-1.7B-Base"
    if local_model.exists():
        return local_model.resolve()
    return config.vllm_omni_model


def _open_audio_file(path: Path) -> None:
    webbrowser.open(path.resolve().as_uri())


def _write_wav_file(path: Path, samples: Any, sample_rate: int) -> None:
    import wave

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())


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


def _merged_detected_options(base_options: List[str], detected_values: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [*base_options, *detected_values]:
        text = str(value).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def _detected_identity_values(paths: BookPaths, registry: Dict[str, Any], key: str) -> List[str]:
    values: List[str] = []
    for character in registry.get("characters", {}).values():
        if not isinstance(character, dict):
            continue
        identity = character.get("identity_profile", {})
        if isinstance(identity, dict) and identity.get(key):
            values.append(str(identity[key]))

    annotations_dir = paths.root / "annotations"
    if not annotations_dir.exists():
        return values
    for annotation_path in sorted(annotations_dir.glob("*.annotation.json")):
        payload = read_json(annotation_path)
        for speaker in payload.get("local_speakers", []):
            profile = speaker.get("profile", {})
            if isinstance(profile, dict) and profile.get(key):
                values.append(str(profile[key]))
        for speaker in payload.get("proposed_new_characters", []):
            profile = speaker.get("profile", {})
            if isinstance(profile, dict) and profile.get(key):
                values.append(str(profile[key]))
    return values


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


def _bounded_positive_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    parsed = _positive_float(value, default)
    return min(float(maximum), max(float(minimum), float(parsed)))


def _choice(value: Any, allowed: set, default: str) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default
