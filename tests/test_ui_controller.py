from pathlib import Path

import pytest

from ebook_tts_pipeline.epub_ingestion import EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.read_along.narrator_profile import narrator_profile_hash
from ebook_tts_pipeline.registry import RegistryManager, build_compact_voice_profile, voice_profile_hash
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
from ebook_tts_pipeline.ui import controller as controller_module
from ebook_tts_pipeline.ui.controller import ChapterStage, PrototypeUiController


class FakeExtractor:
    def extract(self, epub_path, paths):
        paths.chapter_text("chapter_001").parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
        paths.chapter_text("chapter_002").write_text("Chapter Two\nText.", encoding="utf-8")
        return EpubExtractResult(
            chapters=["chapter_001", "chapter_002"],
            sources=["text/chapter001.xhtml", "text/chapter002.xhtml"],
        )


class FakeRegistry:
    def __init__(self, paths, calls):
        self.paths = paths
        self.calls = calls

    def initialize_if_missing(self, book_title, book_slug, book_author=""):
        self.calls.append(("initialize", book_title, book_slug))
        if not self.paths.registry.exists():
            book = {"title": book_title, "slug": book_slug}
            if book_author:
                book["author"] = book_author
            write_json_atomic(
                self.paths.registry,
                {"book": book, "characters": {}},
            )

    def load(self):
        return read_json(self.paths.registry)


class FakePipeline:
    def __init__(self, config, calls):
        self.paths = BookPaths(config.book_root)
        self.calls = calls
        self.config = config
        self.registry = FakeRegistry(self.paths, calls)

    def segment_chapter(self, chapter):
        self.calls.append(("segment", chapter))
        self.paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.sentence_artifact(chapter),
            {"chapter": chapter, "source_path": f"chapters/{chapter}.txt", "sentences": []},
        )

    def annotate_chapter(self, chapter, lock_registry=False):
        self.calls.append(("annotate", chapter, lock_registry))
        self.paths.annotation(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.annotation(chapter),
            {
                "new_characters": [],
                "roles": ["Narrator"],
                "types": ["narration", "dialogue", "thought"],
                "script": [],
            },
        )

    def build_global_registry(self, book_title=None):
        self.calls.append(("build_global_registry", book_title))
        registry = read_json(self.paths.registry)
        registry["characters"] = {
            "akari_adult": {
                "role_id": "akari_adult",
                "display_name": "Akari",
                "aliases": [],
                "identity_profile": {
                    "age": None,
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["professional"],
                    "race_or_ethnicity": None,
                    "accent": None,
                },
            }
        }
        write_json_atomic(self.paths.registry, registry)
        return 1

    def build_sentence_jobs(self, chapter, annotation):
        self.calls.append(("build_scripts", chapter))
        self.paths.tts_script(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.paths.tts_script(chapter), {"chapter": chapter, "jobs": []})
        self.paths.qwen_script(chapter).write_text("Narrator: Text.\n", encoding="utf-8")

    def prepare_voices_for_annotation(self, annotation, chapter=None, include_narrator=True):
        self.calls.append(("prepare_voices", chapter))

    def synthesize_chapter_from_tts_script(self, chapter):
        self.calls.append(("synthesize", chapter))
        self.paths.chapter_audio(chapter).parent.mkdir(parents=True, exist_ok=True)
        self.paths.chapter_audio(chapter).write_bytes(b"wav")


class VoiceAssetPipeline:
    def __init__(self, config) -> None:
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.tts_adapter = FakeTtsAdapter()

    def _voice_path_for_record(self, role_id, record):
        return self.paths.voice_qvp(role_id)


class RecordingFakeTtsAdapter(FakeTtsAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.ensure_voice_calls = []
        self.generated_role_ids = []
        self.generated_texts = []

    def ensure_voice(self, role_id, voice_record, voice_path):
        self.ensure_voice_calls.append((role_id, Path(voice_path)))
        return super().ensure_voice(role_id, voice_record, voice_path)

    def generate_sentences(self, jobs):
        self.generated_role_ids.extend(str(job["role_id"]) for job in jobs)
        self.generated_texts.extend(str(job["text"]) for job in jobs)
        return super().generate_sentences(jobs)


class RecordingVoiceAssetPipeline:
    def __init__(self, config) -> None:
        self.paths = BookPaths(config.book_root)
        self.registry = RegistryManager(self.paths)
        self.tts_adapter = RecordingFakeTtsAdapter()

    def _voice_path_for_record(self, role_id, record):
        return self.paths.voice_qvp(role_id)

    def build_read_along_units(self, chapter):
        return read_json(self.paths.read_along_units(chapter))["units"]


class FakeProgressPipeline(FakePipeline):
    def annotate_chapter(self, chapter, lock_registry=False):
        self.calls.append(("annotate", chapter, lock_registry))
        self.paths.annotation(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.paths.annotation(chapter),
            {"schema": "quote_attribution_v1", "roles": [], "quotes": []},
        )
        return {"schema": "quote_attribution_v1", "roles": [], "quotes": []}

    def build_read_along_units(self, chapter):
        self.calls.append(("build_units", chapter))
        self.paths.read_along_units(chapter).parent.mkdir(parents=True, exist_ok=True)
        payload = {"chapter": chapter, "units": []}
        write_json_atomic(self.paths.read_along_units(chapter), payload)
        return payload["units"]


class FailingChapterPipeline(FakeProgressPipeline):
    def annotate_chapter(self, chapter, lock_registry=False):
        if chapter == "chapter_002":
            raise RuntimeError("model timed out")
        return super().annotate_chapter(chapter, lock_registry=lock_registry)


def fake_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(
            (
                "factory",
                needs_llm,
                fake_tts,
                config.tts_speed,
                config.pause_between_sentences_ms,
                config.intra_sentence_pause_ms,
                config.tts_backend,
            )
        )
        return FakePipeline(config, calls)

    return factory


def fake_progress_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts, config.book_root))
        return FakeProgressPipeline(config, calls)

    return factory


def failing_chapter_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts, config.book_root))
        return FailingChapterPipeline(config, calls)

    return factory


def test_controller_detects_chapter_stage_from_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\n", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    paths.sentence_artifact("chapter_001").write_text("{}", encoding="utf-8")
    controller = PrototypeUiController(book_root=paths.root)

    assert controller.chapter_rows()[0].stage == ChapterStage.SEGMENTED

    paths.annotation("chapter_001").parent.mkdir(parents=True)
    paths.annotation("chapter_001").write_text("{}", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.ANNOTATION_REVIEW

    paths.annotation_approval("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    paths.annotation_approval("chapter_001").write_text('{"approved": true}', encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.ANNOTATED

    paths.tts_script("chapter_001").parent.mkdir(parents=True)
    paths.tts_script("chapter_001").write_text("{}", encoding="utf-8")
    paths.qwen_script("chapter_001").write_text("Narrator: Hi.\n", encoding="utf-8")
    assert controller.chapter_rows()[0].stage == ChapterStage.SCRIPTED

    paths.chapter_audio("chapter_001").parent.mkdir(parents=True)
    paths.chapter_audio("chapter_001").write_bytes(b"wav")
    assert controller.chapter_rows()[0].stage == ChapterStage.AUDIO


def test_controller_saves_pretty_registry_json(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root)

    controller.save_registry_text('{"book":{"title":"Demo"},"characters":{}}')

    text = paths.registry.read_text(encoding="utf-8")
    assert '"title": "Demo"' in text

    with pytest.raises(ValueError):
        controller.save_registry_text("{bad json")


def test_controller_saves_tts_settings_and_applies_them_to_pipeline_config(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    controller.save_tts_settings(
        {
            "tts_speed": "1.25",
            "pause_between_sentences_ms": "150",
            "intra_sentence_pause_ms": "35",
        }
    )
    settings = controller.tts_settings()
    controller.run_next_chapter_action("chapter_001")

    assert settings == {
        "tts_speed": 1.25,
        "pause_between_sentences_ms": 150,
        "intra_sentence_pause_ms": 35,
    }
    assert ("factory", True, True, 1.25, 150, 35, "native") in calls


def test_controller_registry_forms_expose_only_safe_editable_fields(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age": None,
                    "age_stage": "adult",
                    "aliases": ["Callie adult"],
                    "identity_profile": {
                        "age": None,
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["stressed", "protective"],
                        "race_or_ethnicity": None,
                        "accent": None,
                        "occupation": "caregiver",
                    },
                    "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                    "voice_variants": {
                        "default": {
                            "role_id": "callie_adult_default",
                            "display_name": "Callie_default",
                            "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                            "voice_profile": {
                                "description": "adult female; stressed, protective",
                                "qwen_instruct": "A adult female voice.",
                            },
                            "voice_config_path": "voices/callie_adult_default.qvp",
                            "voice_config_hash": "old-hash",
                        }
                    },
                }
            },
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    form = controller.registry_character_forms()[0]
    editable_keys = {field.key for field in form.editable_fields}
    readonly_keys = {field.key for field in form.readonly_fields}

    assert form.role_id == "callie_adult"
    assert "role_id" in readonly_keys
    assert "person_id" in readonly_keys
    assert "seed" in readonly_keys
    assert "display_name" in editable_keys
    assert "age" not in editable_keys
    assert "age_stage" in editable_keys
    assert "personality" in editable_keys
    assert "occupation" in editable_keys
    assert "narrative_notes" not in editable_keys
    assert "role_id" not in editable_keys
    assert "seed" not in editable_keys


def test_controller_saves_registry_form_values_and_refreshes_voice_profile(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age": None,
                    "age_stage": "adult",
                    "aliases": ["Callie adult"],
                    "identity_profile": {
                        "age": None,
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["stressed"],
                        "race_or_ethnicity": None,
                        "accent": None,
                        "occupation": None,
                    },
                    "timeline": "legacy",
                    "same_person_as": ["legacy_callie"],
                    "character_profile": {"gender": "female"},
                    "narrative_notes": "Original notes.",
                    "first_seen": "chapter_001",
                    "global_evidence": [{"chapter": "chapter_001"}],
                    "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                    "voice_variants": {
                        "default": {
                            "role_id": "callie_adult_default",
                            "display_name": "Callie_default",
                            "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                            "voice_profile": {
                                "description": "adult female; stressed",
                                "qwen_instruct": "A adult female voice.",
                            },
                            "voice_config_path": "voices/callie_adult_default.qvp",
                            "voice_config_hash": "old-hash",
                        },
                        "internal": {
                            "role_id": "callie_adult_internal",
                            "display_name": "Callie_internal",
                            "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                            "voice_profile": {
                                "description": "adult female; stressed; internal",
                                "qwen_instruct": "A adult female voice. Internal.",
                            },
                            "voice_config_path": "voices/callie_adult_internal.qvp",
                            "voice_config_hash": "old-internal-hash",
                        },
                    },
                }
            },
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    controller.save_registry_character_form(
        "callie_adult",
        {
            "display_name": "Callie",
            "age": "14",
            "age_stage": "teen",
            "gender": "female",
            "personality": "guarded, timid",
            "race_or_ethnicity": "",
            "accent": "",
            "occupation": "student",
            "aliases": "Callie teen, Callie",
            "narrative_notes": "Victim of grooming/exploitation; not romance.",
        },
    )

    registry = read_json(paths.registry)
    character = registry["characters"]["callie_adult"]
    assert "age" not in character
    assert "age" not in character["identity_profile"]
    assert character["age_stage"] == "teen"
    assert character["display_name"] == "Callie"
    assert character["identity_profile"]["personality"] == ["guarded", "timid"]
    assert character["identity_profile"]["occupation"] == "student"
    assert character["aliases"] == ["Callie teen", "Callie"]
    assert "timeline" not in character
    assert "same_person_as" not in character
    assert "character_profile" not in character
    assert "narrative_notes" not in character
    assert "first_seen" not in character
    assert "global_evidence" not in character
    assert "voice_variants" not in character
    assert "teen female" in character["voice_profile"]["qwen_instruct"]
    assert character.get("voice_config_hash") is None


def test_controller_registry_review_payload_excludes_narrator_and_detects_race_accent_options(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_config_path": "voices/narrator.qvp",
                "voice_config_hash": "narrator-hash",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
            },
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age_stage": "adult",
                    "aliases": ["Callie adult"],
                    "voice_config_path": "voices/callie_adult.qvp",
                    "voice_config_hash": "character-hash",
                    "identity_profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["guarded"],
                        "race_or_ethnicity": "Japanese",
                        "accent": "Tokyo",
                        "occupation": "lawyer",
                    },
                    "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                }
            },
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "schema": "quote_attribution_v1",
            "roles": ["Security Guard", "Visitor"],
            "quotes": [[0, 0], [1, 1]],
            "local_speakers": [
                {
                    "local_id": "tmp_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "race_or_ethnicity": "Brazilian",
                        "accent": "Rio",
                    },
                }
            ],
            "proposed_new_characters": [
                {
                    "name": "Visitor",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "race_or_ethnicity": "Nigerian",
                        "accent": "Lagos",
                    },
                }
            ],
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    payload = controller.registry_review_payload()

    assert payload["book"] == {"title": "Demo", "slug": "demo"}
    assert [entry["role_id"] for entry in payload["entries"]] == ["callie_adult"]
    callie = payload["entries"][0]
    assert callie["kind"] == "character"
    assert callie["editable"] is True
    assert callie["fields"]["display_name"] == "Callie"
    assert callie["fields"]["race_or_ethnicity"] == "Japanese"
    assert callie["fields"]["accent"] == "Tokyo"
    assert callie["voice_config_path"] == "voices/callie_adult.qvp"
    assert "voice_config_hash" not in callie
    assert "qwen_instruct" not in callie
    assert "seed" not in callie
    assert "Tokyo" in payload["accent_options"]
    assert "Rio" in payload["accent_options"]
    assert "Lagos" in payload["accent_options"]
    assert "Yorkshire" in payload["accent_options"]
    assert "Received Pronunciation" in payload["accent_options"]
    assert "French" in payload["accent_options"]
    assert "Japanese" in payload["race_or_ethnicity_options"]
    assert "Brazilian" in payload["race_or_ethnicity_options"]
    assert "Nigerian" in payload["race_or_ethnicity_options"]


def test_compact_voice_profile_expands_british_accent_to_phonetic_constraints():
    profile = build_compact_voice_profile(
        "Winifred",
        {
            "identity_profile": {
                "age_stage": "adult",
                "gender": "female",
                "personality": ["darkly witty"],
                "race_or_ethnicity": "English",
                "accent": "British",
            }
        },
    )

    instruction = profile["qwen_instruct"]
    assert "British English pronunciation" in instruction
    assert "non-rhotic R" in instruction
    assert "no General American" in instruction


def test_compact_voice_profile_expands_received_pronunciation_and_yorkshire():
    rp = build_compact_voice_profile(
        "Mr Pounds",
        {"identity_profile": {"age_stage": "adult", "gender": "male", "accent": "Received Pronunciation"}},
    )["qwen_instruct"]
    yorkshire = build_compact_voice_profile(
        "The Phaeton Driver",
        {"identity_profile": {"age_stage": "adult", "gender": "male", "accent": "Yorkshire"}},
    )["qwen_instruct"]

    assert "upper-class southern British Received Pronunciation" in rp
    assert "clipped precise consonants" in rp
    assert "Yorkshire / Northern English accent" in yorkshire
    assert "northern English vowel shapes" in yorkshire


def test_compact_voice_profile_expands_french_accent():
    profile = build_compact_voice_profile(
        "The French Nurse",
        {"identity_profile": {"age_stage": "adult", "gender": "female", "accent": "French"}},
    )

    assert "French-accented English" in profile["qwen_instruct"]
    assert "softened th consonants" in profile["qwen_instruct"]


def test_controller_registry_review_infers_blank_accent_from_race_background(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "characters": {
                "french_nurse_adult": {
                    "role_id": "french_nurse_adult",
                    "display_name": "The French Nurse",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": [],
                        "race_or_ethnicity": "French",
                        "accent": None,
                    },
                    "voice_identity": {"seed": 123, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                }
            },
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    payload = controller.registry_review_payload()

    assert payload["entries"][0]["fields"]["accent"] == "French"


def test_controller_generates_registry_voice_sample_with_voice_asset_backend(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_config_path": None,
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
                    "voice_identity": {"seed": 3, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": None,
                }
            },
        },
    )
    captured = []
    pipelines = []

    def factory(config, needs_llm, fake_tts):
        captured.append((config.tts_backend, needs_llm, fake_tts))
        pipeline = RecordingVoiceAssetPipeline(config)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setenv("EBOOK_TTS_BACKEND", "wsl-vllm-omni")
    monkeypatch.delenv("EBOOK_TTS_VOICE_ASSET_BACKEND", raising=False)
    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=factory)

    sample = controller.generate_registry_voice_sample("leigh_adult")

    assert captured == [("wsl", False, False)]
    assert sample["role_id"] == "leigh_adult"
    assert sample["sample_url"] == "/api/registry/sample/leigh_adult.wav"
    sample_path = paths.root / sample["sample_path"]
    assert sample_path.read_bytes()[:4] == b"RIFF"
    assert pipelines[0].tts_adapter.generated_texts == [
        "Hello, my name is Leigh. After the party, I asked for a glass of water, "
        "a little butter, and a proper cup of tea."
    ]
    registry = read_json(paths.registry)
    assert registry["characters"]["leigh_adult"]["voice_config_path"] == "voices/leigh_adult.qvp"
    assert registry["characters"]["leigh_adult"]["voice_config_hash"]


def test_controller_annotation_review_uses_registry_age_stage_options(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_callie_registry(paths)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "roles": ["Narrator", "Callie adult"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    forms = controller.annotation_appearance_forms("chapter_001")

    assert len(forms) == 1
    assert forms[0].key == "callie"
    assert forms[0].name == "Callie"
    assert forms[0].current_age_stage == "adult"
    assert [(option.age_stage, option.role_name) for option in forms[0].age_stage_options] == [
        ("adult", "Callie adult"),
        ("child", "Callie child"),
    ]


def test_controller_confirming_annotation_appearance_rewrites_roles_and_approves(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_callie_registry(paths)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "roles": ["Narrator", "Callie adult"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1], [1, 2, 2]],
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    controller.confirm_annotation_appearances("chapter_001", {"callie": "child"})

    annotation = read_json(paths.annotation("chapter_001"))
    approval = read_json(paths.annotation_approval("chapter_001"))
    assert annotation["roles"] == ["Narrator", "Callie child"]
    assert annotation["script"] == [[0, 0, 0], [1, 1, 1], [1, 2, 2]]
    assert approval["approved"] is True
    assert approval["appearances"] == [
        {
            "person_id": "callie",
            "name": "Callie",
            "age_stage": "child",
            "role_name": "Callie child",
        }
    ]
    assert controller.chapter_stage("chapter_001") == ChapterStage.ANNOTATED


def test_controller_confirming_quote_annotation_preserves_quote_mapping(tmp_path):
    paths = BookPaths(tmp_path / "book")
    _write_callie_registry(paths)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "schema": "quote_attribution_v1",
            "roles": ["callie_adult"],
            "quotes": [[1, 0]],
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    controller.confirm_annotation_appearances("chapter_001", {"callie": "child"})

    annotation = read_json(paths.annotation("chapter_001"))
    assert annotation["schema"] == "quote_attribution_v1"
    assert annotation["roles"] == ["Callie child"]
    assert annotation["quotes"] == [[1, 0]]
    assert "script" not in annotation
    assert "types" not in annotation


def test_controller_blocks_script_generation_until_annotation_is_approved(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.sentence_artifact("chapter_001"),
        {"chapter": "chapter_001", "source_path": "chapters/chapter_001.txt", "sentences": []},
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "roles": ["Narrator"],
            "types": ["narration", "dialogue", "thought"],
            "script": [],
        },
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    blocked = controller.run_next_chapter_action("chapter_001")
    assert blocked.stage == ChapterStage.ANNOTATION_REVIEW
    assert "Review" in blocked.message
    assert ("build_scripts", "chapter_001") not in calls

    paths.annotation_approval("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.annotation_approval("chapter_001"), {"approved": True, "appearances": []})
    scripted = controller.run_next_chapter_action("chapter_001")

    assert scripted.stage == ChapterStage.SCRIPTED
    assert ("build_scripts", "chapter_001") in calls


def test_controller_converts_proposed_characters_to_local_speakers_when_appearances_are_confirmed(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {},
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "proposed_new_characters": [
                {
                    "name": "Houseless man",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["anxious", "paranoid"],
                    },
                }
            ],
            "roles": ["Narrator", "Houseless man"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        },
    )
    controller = PrototypeUiController(book_root=paths.root)

    controller.confirm_annotation_appearances("chapter_001", {})

    registry = read_json(paths.registry)
    annotation = read_json(paths.annotation("chapter_001"))
    assert registry["characters"] == {}
    assert annotation.get("proposed_new_characters", []) == []
    assert annotation["local_speakers"] == [
        {
            "local_id": "tmp_001",
            "label": "Houseless man",
            "profile": {
                "age_stage": "adult",
                "gender": "male",
                "personality": ["anxious", "paranoid"],
            },
        }
    ]
    assert read_json(paths.annotation_approval("chapter_001"))["approved"] is True


def test_controller_converts_proposed_characters_to_local_speakers_before_script_generation(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {},
        },
    )
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.sentence_artifact("chapter_001"),
        {"chapter": "chapter_001", "source_path": "chapters/chapter_001.txt", "sentences": []},
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "new_characters": [],
            "proposed_new_characters": [
                {
                    "name": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["authoritative"],
                        "occupation": "security guard",
                    },
                }
            ],
            "roles": ["Narrator", "Security Guard"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        },
    )
    paths.annotation_approval("chapter_001").parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.annotation_approval("chapter_001"), {"approved": True, "appearances": []})
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    result = controller.run_next_chapter_action("chapter_001")

    registry = read_json(paths.registry)
    annotation = read_json(paths.annotation("chapter_001"))
    assert result.stage == ChapterStage.SCRIPTED
    assert registry["characters"] == {}
    assert annotation.get("proposed_new_characters", []) == []
    assert annotation["local_speakers"] == [
        {
            "local_id": "tmp_001",
            "label": "Security Guard",
            "profile": {
                "age_stage": "adult",
                "gender": "male",
                "personality": ["authoritative"],
                "occupation": "security guard",
            },
        }
    ]
    assert ("build_scripts", "chapter_001") in calls


def test_controller_loads_epub_initializes_registry_segments_and_toc(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    library_path = tmp_path / "library.json"
    controller = PrototypeUiController(
        book_root=paths.root,
        library_path=library_path,
        pipeline_factory=fake_pipeline_factory(calls),
        extractor=FakeExtractor(),
        fake_tts=True,
    )

    result = controller.load_epub(tmp_path / "demo.epub", title="Demo", slug="demo")

    assert result.chapters == ["chapter_001", "chapter_002"]
    assert paths.registry.exists()
    assert paths.sentence_artifact("chapter_001").exists()
    assert paths.sentence_artifact("chapter_002").exists()
    toc = paths.root / "toc.json"
    assert "Chapter One" in toc.read_text(encoding="utf-8")
    library = controller.library_books()
    assert len(library) == 1
    assert library[0].title == "Demo"
    assert library[0].slug == "demo"
    assert library[0].book_root == paths.root
    assert library[0].epub_path == tmp_path / "demo.epub"
    assert ("initialize", "Demo", "demo") in calls
    assert ("segment", "chapter_001") in calls
    assert ("segment", "chapter_002") in calls


def test_controller_switches_between_library_books(tmp_path):
    first = BookPaths(tmp_path / "first")
    second = BookPaths(tmp_path / "second")
    first.chapter_text("chapter_001").parent.mkdir(parents=True)
    first.chapter_text("chapter_001").write_text("First Book\n", encoding="utf-8")
    second.chapter_text("chapter_001").parent.mkdir(parents=True)
    second.chapter_text("chapter_001").write_text("Second Book\n", encoding="utf-8")
    library_path = tmp_path / "library.json"
    write_json_atomic(
        library_path,
        {
            "books": [
                {
                    "title": "First",
                    "slug": "first",
                    "book_root": str(first.root),
                    "epub_path": str(tmp_path / "first.epub"),
                },
                {
                    "title": "Second",
                    "slug": "second",
                    "book_root": str(second.root),
                    "epub_path": str(tmp_path / "second.epub"),
                },
            ]
        },
    )
    controller = PrototypeUiController(book_root=first.root, library_path=library_path)

    controller.select_book("second")

    assert controller.book_root == second.root
    assert controller.current_book_slug == "second"
    assert controller.chapter_rows()[0].title == "Second Book"


def test_controller_chapter_action_advances_through_pipeline_stages(tmp_path):
    calls = []
    opened = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    paths.sentence_artifact("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.sentence_artifact("chapter_001"),
        {"chapter": "chapter_001", "source_path": "chapters/chapter_001.txt", "sentences": []},
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        audio_opener=opened.append,
        fake_tts=True,
    )

    first = controller.run_next_chapter_action("chapter_001")
    controller.confirm_annotation_appearances("chapter_001", {})
    second = controller.run_next_chapter_action("chapter_001")
    third = controller.run_next_chapter_action("chapter_001")
    fourth = controller.run_next_chapter_action("chapter_001")

    assert first.stage == ChapterStage.ANNOTATION_REVIEW
    assert second.stage == ChapterStage.SCRIPTED
    assert third.stage == ChapterStage.AUDIO
    assert fourth.stage == ChapterStage.AUDIO
    assert opened == [paths.chapter_audio("chapter_001")]
    assert ("annotate", "chapter_001", True) in calls
    assert ("build_scripts", "chapter_001") in calls
    assert ("prepare_voices", "chapter_001") in calls
    assert ("synthesize", "chapter_001") in calls


def test_controller_builds_global_registry(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {"book": {"title": "Demo", "slug": "demo"}, "characters": {}},
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    count = controller.build_global_registry()

    assert count == 1
    assert ("build_global_registry", "Demo") in calls
    assert read_json(paths.registry)["characters"]["akari_adult"]["display_name"] == "Akari"


def _write_callie_registry(paths):
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator"},
            "characters": {
                "callie_adult": {
                    "role_id": "callie_adult",
                    "profile_id": "callie_adult",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age_stage": "adult",
                    "aliases": ["Callie adult"],
                    "identity_profile": {
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["guarded"],
                    },
                    "voice_identity": {"seed": 1, "differentiators": []},
                    "voice_variants": {},
                },
                "callie_child": {
                    "role_id": "callie_child",
                    "profile_id": "callie_child",
                    "person_id": "callie",
                    "display_name": "Callie",
                    "age_stage": "child",
                    "aliases": ["Callie child"],
                    "identity_profile": {
                        "age_stage": "child",
                        "gender": "female",
                        "personality": ["trusting"],
                    },
                    "voice_identity": {"seed": 2, "differentiators": []},
                    "voice_variants": {},
                },
            },
        },
    )


def test_controller_builds_global_registry_initializes_missing_registry(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    library_path = tmp_path / "library.json"
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Chapter One\nText.", encoding="utf-8")
    write_json_atomic(
        library_path,
        {
            "books": [
                {
                    "title": "Demo",
                    "slug": "demo",
                    "book_root": str(paths.root),
                    "epub_path": str(tmp_path / "demo.epub"),
                }
            ]
        },
    )
    controller = PrototypeUiController(
        book_root=paths.root,
        library_path=library_path,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=True,
    )

    count = controller.build_global_registry()

    assert count == 1
    assert ("initialize", "Demo", "demo") in calls
    assert ("build_global_registry", "Demo") in calls
    assert read_json(paths.registry)["book"] == {"title": "Demo", "slug": "demo"}
    assert read_json(paths.registry)["characters"]["akari_adult"]["display_name"] == "Akari"


def test_controller_builds_read_along_units_from_quote_annotation(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    registry = read_json(paths.registry)
    registry["narrator"]["voice_config_path"] = "voices/narrator.qvp"
    registry["narrator"]["voice_profile"] = {"description": "male narrator", "qwen_instruct": "male narrator"}
    registry["characters"]["leigh_adult"] = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "voice_config_path": "voices/leigh_adult.qvp",
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_identity": {"seed": 3, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
    }
    write_json_atomic(paths.registry, registry)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    units = controller.build_read_along_units("chapter_001")

    assert paths.read_along_units("chapter_001").exists()
    assert any(unit["role_id"] == "leigh_adult" for unit in units)
    assert any(unit["role_id"] == "narrator" for unit in units)


def test_controller_prepare_read_along_voices_prepares_global_only_and_defers_local_temp_speakers(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Narration. "Stop there," the guard said.', encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_config_path": None,
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
                    "voice_identity": {"seed": 2, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": None,
                }
            },
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "schema": "quote_attribution_v1",
            "roles": ["Security Guard"],
            "quotes": [[1, 0]],
            "local_speakers": [
                {
                    "local_id": "tmp_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["authoritative"],
                        "occupation": "security guard",
                    },
                }
            ],
        },
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    result = controller.prepare_read_along_voices()

    assert result["chapters"] == 1
    assert result["voices_ready"] is True
    assert not paths.voice_qvp("narrator").exists()
    assert not (paths.root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp").exists()
    assert (paths.root / "voices" / "_samples" / "leigh_adult.wav").exists()
    units = read_json(paths.read_along_units("chapter_001"))["units"]
    assert any(unit["voice_config_path"] == "voices/_temp/chapter_001/tmp_001.qvp" for unit in units)
    assert any(unit["role_id"] == "narrator" and unit["voice_config_path"] is None for unit in units)


def test_controller_prepare_read_along_voices_skips_tts_pipeline_when_registry_assets_cached(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Hello."', encoding="utf-8")
    character = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
        "voice_identity": {"seed": 2, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
        "voice_config_path": "voices/leigh_adult.qvp",
    }
    character["voice_config_hash"] = voice_profile_hash(character)
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator", "voice_config_path": None},
            "characters": {"leigh_adult": character},
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    paths.read_along_units("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.read_along_units("chapter_001"),
        {
            "chapter": "chapter_001",
            "units": [
                {
                    "chapter": "chapter_001",
                    "unit_id": 0,
                    "text": '"Hello."',
                    "source_start": 12,
                    "source_end": 20,
                    "role": "Leigh",
                    "role_id": "leigh_adult",
                    "type": "dialogue",
                    "voice_config_path": "voices/leigh_adult.qvp",
                    "quote_id": "q001",
                    "sentence_idx": 0,
                    "character": "Leigh",
                    "voice_variant": None,
                }
            ],
        },
    )
    paths.voice_qvp("leigh_adult").parent.mkdir(parents=True, exist_ok=True)
    paths.voice_qvp("leigh_adult").write_bytes(b"cached qvp")
    sample_path = paths.root / "voices" / "_samples" / "leigh_adult.wav"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"cached sample")
    calls = []

    def fail_if_pipeline_created(config, needs_llm, fake_tts):
        calls.append((needs_llm, fake_tts))
        raise AssertionError("cached voice generation should not construct a TTS pipeline")

    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=fail_if_pipeline_created, fake_tts=True)

    result = controller.prepare_read_along_voices()

    assert calls == []
    assert result["sample_count"] == 0
    assert result["prepared_chapters"] == 0
    assert result["voice_count"] == 1
    assert result["voice_total"] == 1
    assert result["voices_ready"] is True


def test_controller_prepare_read_along_voices_generates_only_missing_samples(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('"Hi." "There."', encoding="utf-8")
    characters = {}
    for role_id, display_name in [("leigh_adult", "Leigh"), ("callie_adult", "Callie")]:
        record = {
            "role_id": role_id,
            "profile_id": role_id,
            "person_id": role_id.split("_")[0],
            "display_name": display_name,
            "age_stage": "adult",
            "aliases": [],
            "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
            "voice_identity": {"seed": 2, "differentiators": []},
            "voice_profile": {"description": f"{display_name} adult female", "qwen_instruct": f"{display_name} adult female"},
            "voice_config_path": f"voices/{role_id}.qvp",
        }
        record["voice_config_hash"] = voice_profile_hash(record)
        characters[role_id] = record
        paths.voice_qvp(role_id).parent.mkdir(parents=True, exist_ok=True)
        paths.voice_qvp(role_id).write_bytes(f"cached {role_id}".encode("utf-8"))
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"role_id": "narrator", "display_name": "Narrator", "voice_config_path": None},
            "characters": characters,
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult", "callie_adult"], "quotes": [[0, 0], [1, 1]]},
    )
    paths.read_along_units("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.read_along_units("chapter_001"),
        {
            "chapter": "chapter_001",
            "units": [
                {
                    "chapter": "chapter_001",
                    "unit_id": 0,
                    "text": '"Hi."',
                    "source_start": 0,
                    "source_end": 5,
                    "role": "Leigh",
                    "role_id": "leigh_adult",
                    "type": "dialogue",
                    "voice_config_path": "voices/leigh_adult.qvp",
                    "quote_id": "q001",
                    "sentence_idx": 0,
                    "character": "Leigh",
                    "voice_variant": None,
                },
                {
                    "chapter": "chapter_001",
                    "unit_id": 1,
                    "text": '"There."',
                    "source_start": 6,
                    "source_end": 14,
                    "role": "Callie",
                    "role_id": "callie_adult",
                    "type": "dialogue",
                    "voice_config_path": "voices/callie_adult.qvp",
                    "quote_id": "q002",
                    "sentence_idx": 1,
                    "character": "Callie",
                    "voice_variant": None,
                },
            ],
        },
    )
    sample_dir = paths.root / "voices" / "_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "leigh_adult.wav").write_bytes(b"cached sample")
    pipelines = []

    def factory(config, needs_llm, fake_tts):
        pipeline = RecordingVoiceAssetPipeline(config)
        pipelines.append(pipeline)
        return pipeline

    controller = PrototypeUiController(book_root=paths.root, pipeline_factory=factory, fake_tts=True)

    result = controller.prepare_read_along_voices()

    assert len(pipelines) == 1
    assert pipelines[0].tts_adapter.ensure_voice_calls == []
    assert pipelines[0].tts_adapter.generated_role_ids == ["callie_adult"]
    assert result["sample_count"] == 1
    assert result["voice_count"] == 2
    assert result["voice_total"] == 2
    assert result["voices_ready"] is True


def test_controller_read_along_session_generates_local_temp_voice_at_session_start(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Narration. "Stop there," the guard said.', encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "male narrator", "qwen_instruct": "male narrator"},
                "voice_config_path": None,
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": ["direct"]},
                    "voice_identity": {"seed": 2, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": None,
                }
            },
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {
            "schema": "quote_attribution_v1",
            "roles": ["Security Guard"],
            "quotes": [[1, 0]],
            "local_speakers": [
                {
                    "local_id": "tmp_001",
                    "label": "Security Guard",
                    "profile": {
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["authoritative"],
                        "occupation": "security guard",
                    },
                }
            ],
        },
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    controller.prepare_read_along_voices()
    units = controller.read_along_units("chapter_001")
    temp_path = paths.root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp"
    if temp_path.exists():
        temp_path.unlink()

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )

    assert temp_path.exists()
    patched_temp = [unit for unit in session.units if unit.role_id == "chapter_001_tmp_001"][0]
    assert patched_temp.voice_config_path == "voices/_temp/chapter_001/tmp_001.qvp"
    session.end()


def test_controller_annotate_read_along_book_reports_per_chapter_progress(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    for chapter in ["chapter_001", "chapter_002"]:
        paths.chapter_text(chapter).parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text(chapter).write_text(f"{chapter} text.", encoding="utf-8")
        paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            paths.sentence_artifact(chapter),
            {
                "chapter": chapter,
                "source_path": f"chapters/{chapter}.txt",
                "segmenter": {"name": "test"},
                "sentences": [{"idx": 0, "text": f"{chapter} text."}],
            },
        )
    write_json_atomic(
        paths.root / "toc.json",
        {
            "chapters": [
                {"index": 1, "chapter": "chapter_001", "title": "One", "source": "chapter_001.txt"},
                {"index": 2, "chapter": "chapter_002", "title": "Two", "source": "chapter_002.txt"},
            ]
        },
    )
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    progress = []
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_progress_pipeline_factory(calls),
        fake_tts=True,
    )

    result = controller.annotate_read_along_book(progress_callback=progress.append)

    assert result == {"chapters": 2, "annotated": 2, "units_built": 2}
    assert progress == [
        {"chapter": "chapter_001", "index": 1, "total": 2, "status": "started"},
        {"chapter": "chapter_001", "index": 1, "total": 2, "status": "completed"},
        {"chapter": "chapter_002", "index": 2, "total": 2, "status": "started"},
        {"chapter": "chapter_002", "index": 2, "total": 2, "status": "completed"},
    ]
    progress_file = read_json(paths.root / "read_along" / "annotation_progress.json")
    assert progress_file["status"] == "completed"
    assert progress_file["completed"] == 2
    assert progress_file["total"] == 2


def test_controller_annotate_read_along_book_retries_stale_quote_annotations(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    chapter = "chapter_001"
    paths.chapter_text(chapter).parent.mkdir(parents=True, exist_ok=True)
    paths.chapter_text(chapter).write_text(
        "The driver said, \u2018That there\u2019s Ensor House, there,\u2019 then waited.",
        encoding="utf-8",
    )
    paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.sentence_artifact(chapter),
        {
            "chapter": chapter,
            "source_path": f"chapters/{chapter}.txt",
            "segmenter": {"name": "test"},
            "sentences": [{"idx": 0, "text": "The driver spoke."}],
        },
    )
    write_json_atomic(paths.annotation(chapter), {"schema": "quote_attribution_v1", "roles": [], "quotes": []})
    write_json_atomic(paths.read_along_units(chapter), {"chapter": chapter, "units": []})
    write_json_atomic(
        paths.root / "toc.json",
        {"chapters": [{"index": 1, "chapter": chapter, "title": "One", "source": "chapter_001.txt"}]},
    )
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_progress_pipeline_factory(calls),
        fake_tts=True,
    )

    result = controller.annotate_read_along_book()

    assert result == {"chapters": 1, "annotated": 1, "units_built": 1}
    assert ("annotate", chapter, True) in calls


def test_controller_annotate_read_along_book_reports_failed_chapter(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    for chapter in ["chapter_001", "chapter_002"]:
        paths.chapter_text(chapter).parent.mkdir(parents=True, exist_ok=True)
        paths.chapter_text(chapter).write_text(f"{chapter} text.", encoding="utf-8")
        paths.sentence_artifact(chapter).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            paths.sentence_artifact(chapter),
            {
                "chapter": chapter,
                "source_path": f"chapters/{chapter}.txt",
                "segmenter": {"name": "test"},
                "sentences": [{"idx": 0, "text": f"{chapter} text."}],
            },
        )
    write_json_atomic(
        paths.root / "toc.json",
        {
            "chapters": [
                {"index": 1, "chapter": "chapter_001", "title": "One", "source": "chapter_001.txt"},
                {"index": 2, "chapter": "chapter_002", "title": "Two", "source": "chapter_002.txt"},
            ]
        },
    )
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    progress = []
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=failing_chapter_pipeline_factory(calls),
        fake_tts=True,
    )

    with pytest.raises(RuntimeError, match="Annotation failed at chapter_002: model timed out"):
        controller.annotate_read_along_book(progress_callback=progress.append)

    assert progress[-1] == {
        "chapter": "chapter_002",
        "index": 2,
        "total": 2,
        "status": "failed",
        "error": "model timed out",
    }
    progress_file = read_json(paths.root / "read_along" / "annotation_progress.json")
    assert progress_file["status"] == "failed"
    assert progress_file["failed_chapter"] == "chapter_002"
    assert progress_file["error"] == "model timed out"


def test_read_along_session_requires_prepared_voice_paths(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    registry = read_json(paths.registry)
    registry["narrator"]["voice_profile"] = {
        "description": "adult male narrator",
        "qwen_instruct": "adult male narrator",
    }
    registry["characters"]["leigh_adult"] = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_identity": {"seed": 3, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
        "voice_config_path": None,
    }
    write_json_atomic(paths.registry, registry)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    units = controller.build_read_along_units("chapter_001")

    with pytest.raises(ValueError, match="Prepare Voices"):
        controller.create_read_along_session(
            "chapter_001",
            units,
            {
                "playback_speed": 1.0,
                "generation_mode": "balanced",
                "buffer_limit": 2,
                "target_buffer_seconds": 20,
                "start_buffer_seconds": 20,
                "max_buffer_seconds": 40,
                "max_buffer_units": 32,
            },
        )


def test_controller_read_along_session_uses_prepared_voices_before_buffering(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    registry = read_json(paths.registry)
    registry["narrator"]["voice_profile"] = {
        "description": "adult male narrator",
        "qwen_instruct": "adult male narrator",
    }
    registry["characters"]["leigh_adult"] = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_identity": {"seed": 3, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
    }
    write_json_atomic(paths.registry, registry)
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    controller.save_read_along_settings(
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        }
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    controller.prepare_read_along_voices()
    units = controller.read_along_units("chapter_001")

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )
    buffered = session.fill_buffer(start_unit_id=0)

    profile = controller.read_along_narrator_profile()
    narrator_hash = narrator_profile_hash(profile)
    assert paths.narrator_voice_qvp(narrator_hash, "narrator").exists()
    assert paths.voice_qvp("leigh_adult").exists()
    assert len(buffered) == 2
    session.end()


def test_controller_reuses_cached_narrator_voice_when_profile_hash_unchanged(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Leigh waited.", encoding="utf-8")
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": [], "quotes": []},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    profile = controller.read_along_narrator_profile()
    profile_hash = narrator_profile_hash(profile)
    narrator_path = paths.narrator_voice_qvp(profile_hash, "narrator")
    narrator_path.parent.mkdir(parents=True, exist_ok=True)
    narrator_path.write_bytes(b"cached narrator")
    calls = []

    def fail_if_called(self, role_id, voice_record, voice_path):
        calls.append((role_id, Path(voice_path)))
        raise AssertionError("narrator cache should be reused")

    monkeypatch.setattr(FakeTtsAdapter, "ensure_voice", fail_if_called)
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )

    assert calls == []
    assert [unit.voice_config_path for unit in session.units] == [narrator_path.relative_to(paths.root).as_posix()]
    session.end()


def test_controller_reuses_cached_session_narrator_voice_after_noop_profile_save(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('The phone said, "Please hang up."', encoding="utf-8")
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["Narrator"], "quotes": [[1, 0, "narrator_quote"]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    profile = controller.read_along_narrator_profile()
    identity = profile["identity_profile"]
    narrator_hash = narrator_profile_hash(profile)
    narrator_path = paths.narrator_voice_qvp(narrator_hash, "narrator")
    narrator_path.parent.mkdir(parents=True, exist_ok=True)
    narrator_path.write_bytes(b"cached narrator")

    saved_profile = controller.save_read_along_narrator_profile(
        {
            "display_name": profile["display_name"],
            "age_stage": identity["age_stage"],
            "gender": identity["gender"],
            "personality": ", ".join(identity["personality"]),
            "accent": identity.get("accent") or "",
            "race_or_ethnicity": identity.get("race_or_ethnicity") or "",
            "occupation": identity.get("occupation") or "",
        }
    )

    assert narrator_profile_hash(saved_profile) == narrator_hash

    def fail_if_called(self, role_id, voice_record, voice_path):
        raise AssertionError(f"{role_id} cache should be reused")

    monkeypatch.setattr(FakeTtsAdapter, "ensure_voice", fail_if_called)
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )

    patched_quote = [unit for unit in session.units if unit.quote_id == "q001"][0]
    assert patched_quote.role_id == "narrator"
    assert patched_quote.voice_config_path == narrator_path.relative_to(paths.root).as_posix()
    session.end()


def test_controller_maps_legacy_functional_narrator_units_to_narrator_voice(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    patched = controller._apply_session_narrator_voice_paths(
        [
            {
                "chapter": "chapter_001",
                "unit_id": 0,
                "text": '"Closed"',
                "source_start": 0,
                "source_end": 8,
                "role": "Functional Narrator",
                "role_id": "functional_narrator",
                "type": "narration",
                "voice_config_path": None,
                "quote_id": "q001",
                "sentence_idx": 0,
                "character": None,
                "voice_variant": "functional_narrator",
            }
        ],
        {"narrator": "voices/_narrator/hash/narrator.qvp"},
    )

    assert patched[0]["role"] == "Narrator"
    assert patched[0]["role_id"] == "narrator"
    assert patched[0]["voice_variant"] is None
    assert patched[0]["voice_config_path"] == "voices/_narrator/hash/narrator.qvp"


def test_controller_regenerates_narrator_voice_when_profile_changes(tmp_path, monkeypatch):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text("Leigh waited.", encoding="utf-8")
    write_json_atomic(paths.registry, {"book": {"title": "Demo", "slug": "demo"}, "characters": {}})
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": [], "quotes": []},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    controller.save_read_along_narrator_profile(
        {
            "display_name": "Narrator",
            "age_stage": "adult",
            "gender": "female",
            "personality": "warm, steady",
            "accent": "American",
            "race_or_ethnicity": "",
            "occupation": "audiobook narrator",
        }
    )
    calls = []
    original_ensure_voice = FakeTtsAdapter.ensure_voice

    def recording_ensure_voice(self, role_id, voice_record, voice_path):
        calls.append((role_id, dict(voice_record), Path(voice_path)))
        return original_ensure_voice(self, role_id, voice_record, voice_path)

    monkeypatch.setattr(FakeTtsAdapter, "ensure_voice", recording_ensure_voice)
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )

    narrator_calls = [call for call in calls if call[0] == "narrator"]
    assert narrator_calls
    assert "female" in narrator_calls[-1][1]["voice_profile"]["description"]
    assert "American accent" in narrator_calls[-1][1]["voice_profile"]["description"]
    assert narrator_calls[-1][2].as_posix().endswith(
        "/voices/_narrator/" + narrator_profile_hash(controller.read_along_narrator_profile()) + "/narrator.qvp"
    )
    session.end()


def test_controller_expands_simple_narrator_accent_into_strict_voice_instruction(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    profile = controller.save_read_along_narrator_profile(
        {
            "display_name": "Narrator",
            "age_stage": "adult",
            "gender": "male",
            "personality": "calm, clear, measured",
            "accent": "General American",
            "race_or_ethnicity": "White",
            "occupation": "audiobook narrator",
        }
    )

    identity = profile["identity_profile"]
    instruction = profile["voice_profile"]["qwen_instruct"]

    assert identity["accent"] == "General American"
    assert identity["race_or_ethnicity"] == "White"
    assert "General American" in instruction
    assert "Do not use British" in instruction
    assert "Do not infer a regional accent from race or ethnicity" in instruction
    assert "accent is controlled only by the selected accent field" in instruction
    assert "Maintain the exact same adult male" in instruction
    assert "No accent drift" in instruction
    assert "no character voice switching" in instruction


def test_controller_read_along_session_uses_narrator_voice_for_narrator_quotes(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('The phone said, "Please hang up."', encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "adult male narrator", "qwen_instruct": "adult male narrator"},
                "voice_config_path": None,
            },
            "characters": {
                "leigh_adult": {
                    "role_id": "leigh_adult",
                    "profile_id": "leigh_adult",
                    "person_id": "leigh",
                    "display_name": "Leigh",
                    "age_stage": "adult",
                    "aliases": [],
                    "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
                    "voice_identity": {"seed": 3, "differentiators": []},
                    "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
                    "voice_config_path": "voices/leigh_adult.qvp",
                }
            },
        },
    )
    (paths.root / "voices").mkdir(parents=True, exist_ok=True)
    (paths.root / "voices" / "leigh_adult.qvp").write_bytes(b"voice")
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0, "narrator_quote"]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    units = controller.build_read_along_units("chapter_001")
    quote_unit = [unit for unit in units if unit["quote_id"] == "q001"][0]
    assert quote_unit["role_id"] == "narrator"
    assert quote_unit["voice_variant"] is None
    assert quote_unit["voice_config_path"] is None

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
    )

    profile = controller.read_along_narrator_profile()
    narrator_hash = narrator_profile_hash(profile)
    narrator_path = paths.narrator_voice_qvp(narrator_hash, "narrator")
    assert narrator_path.exists()
    patched_quote = [unit for unit in session.units if unit.quote_id == "q001"][0]
    assert patched_quote.role_id == "narrator"
    assert patched_quote.voice_config_path == narrator_path.relative_to(paths.root).as_posix()
    session.end()


def test_controller_reports_read_along_session_start_setup_progress(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('The phone said, "Please hang up."', encoding="utf-8")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {"description": "adult male narrator", "qwen_instruct": "adult male narrator"},
                "voice_config_path": None,
            },
            "characters": {},
        },
    )
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["Narrator"], "quotes": [[1, 0, "narrator_quote"]]},
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    progress = []
    units = controller.build_read_along_units("chapter_001")

    session = controller.create_read_along_session(
        "chapter_001",
        units,
        {
            "playback_speed": 1.0,
            "generation_mode": "balanced",
            "buffer_limit": 2,
            "target_buffer_seconds": 20,
            "start_buffer_seconds": 20,
            "max_buffer_seconds": 40,
            "max_buffer_units": 32,
        },
        progress_callback=progress.append,
    )

    assert [event["stage"] for event in progress] == [
        "loading_tts_model",
        "building_read_along_units",
        "preparing_narrator_voice",
        "checking_local_chapter_voices",
        "validating_voice_paths",
    ]
    assert progress[0]["message"] == "Loading read-along TTS model."
    assert progress[2]["message"] == "Preparing narrator voice."
    session.end()


def test_controller_saves_read_along_settings_without_narrator_voice_type(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    controller.save_read_along_settings(
        {
            "playback_speed": "1.25",
            "generation_mode": "fast",
            "buffer_limit": "2",
        }
    )

    settings = controller.read_along_settings()
    assert settings["playback_speed"] == 1.25
    assert settings["generation_mode"] == "fast"
    assert settings["buffer_limit"] == 2
    assert "narrator_voice_type" not in settings


def test_controller_read_along_narrator_profile_defaults_to_editable_profile(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    profile = controller.read_along_narrator_profile()

    assert profile["role_id"] == "narrator"
    assert profile["display_name"] == "Narrator"
    assert profile["identity_profile"]["age_stage"] == "adult"
    assert profile["identity_profile"]["gender"] == "male"
    assert "audiobook narrator" in profile["identity_profile"]["occupation"]
    assert profile["voice_profile"]["description"]
    assert paths.read_along_narrator_profile.exists()


def test_controller_migrates_narrator_profile_from_legacy_registry(tmp_path):
    paths = BookPaths(tmp_path / "book")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Story Reader",
                "voice_identity": {"seed": 9, "differentiators": ["warm tone"]},
                "voice_profile": {
                    "description": "warm adult female narrator",
                    "qwen_instruct": "A warm adult female narrator.",
                },
            },
            "characters": {},
        },
    )
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    profile = controller.read_along_narrator_profile()

    assert profile["display_name"] == "Story Reader"
    assert profile["voice_identity"]["seed"] == 9
    assert "warm adult female narrator" in profile["voice_profile"]["description"]
    assert paths.read_along_narrator_profile.exists()


def test_controller_read_along_defaults_are_safe_for_vllm_omni_profile(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    settings = controller.read_along_settings()

    assert settings["playback_speed"] == 1.0
    assert settings["generation_mode"] == "balanced"
    assert settings["buffer_limit"] == 2
    assert settings["target_buffer_seconds"] == 20.0
    assert settings["max_buffer_seconds"] == 40.0
    assert settings["chapter_end_behavior"] == "stop"
    assert "start_buffer_seconds" not in settings


def test_controller_read_along_settings_clamp_speed_to_supported_range(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    controller.save_read_along_settings(
        {
            "playback_speed": "5.5",
            "generation_mode": "fast",
            "buffer_limit": "2",
            "target_buffer_seconds": "20",
            "start_buffer_seconds": "20",
            "max_buffer_seconds": "40",
        }
    )

    settings = controller.read_along_settings()
    assert settings["playback_speed"] == 4.0


def test_controller_read_along_pipeline_uses_vllm_omni_backend_by_default(tmp_path):
    calls = []
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(
        book_root=paths.root,
        pipeline_factory=fake_pipeline_factory(calls),
        fake_tts=False,
    )

    controller._pipeline(needs_llm=False, read_along=True)

    assert any(call[0] == "factory" and call[6] == "wsl-vllm-omni" for call in calls)


def test_controller_resolves_default_vllm_omni_model_to_local_qwen_folder(tmp_path):
    model_root = tmp_path / "models" / "qwen-tts"
    base_model = model_root / "Qwen3-TTS-12Hz-1.7B-Base"
    base_model.mkdir(parents=True)
    config = PipelineConfig(book_root=str(tmp_path / "book"), qwen_model_root=str(model_root))

    resolved = controller_module._resolve_vllm_omni_model(config)

    assert resolved == base_model


def test_controller_resolves_relative_vllm_omni_model_to_absolute_local_folder(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "qwen-tts"
    base_model = model_root / "Qwen3-TTS-12Hz-1.7B-Base"
    base_model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    config = PipelineConfig(book_root="book", qwen_model_root="models/qwen-tts")

    resolved = controller_module._resolve_vllm_omni_model(config)

    assert resolved == base_model.resolve()


def test_controller_wsl_voice_asset_adapter_receives_absolute_model_root(tmp_path, monkeypatch, capsys):
    captured = {}

    class DummyWslAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(controller_module, "WslQwenWorkerAdapter", DummyWslAdapter)
    config = PipelineConfig(
        book_root=str(tmp_path / "book"),
        tts_backend="wsl",
        qwen_model_root="models/qwen-tts",
    )

    controller_module._build_qwen_adapter(config)

    assert Path(captured["model_root"]).is_absolute()
    output = capsys.readouterr().out
    assert "[ebook-tts] build_tts_adapter" in output
    assert "backend=wsl" in output
    assert str(captured["model_root"]) in output


def test_controller_vllm_adapter_receives_absolute_voice_model_root(tmp_path, monkeypatch, capsys):
    captured = {}

    class DummyVllmAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(controller_module, "WslVllmOmniQwenAdapter", DummyVllmAdapter)
    config = PipelineConfig(
        book_root=str(tmp_path / "book"),
        tts_backend="wsl-vllm-omni",
        qwen_model_root="models/qwen-tts",
    )

    controller_module._build_qwen_adapter(config)

    assert Path(captured["voice_model_root"]).is_absolute()
    output = capsys.readouterr().out
    assert "[ebook-tts] build_tts_adapter" in output
    assert "backend=wsl-vllm-omni" in output
    assert str(captured["voice_model_root"]) in output


def test_default_pipeline_factory_delays_qwen_adapter_until_tts_use(tmp_path, monkeypatch):
    calls = []

    class DummyAdapter:
        def ensure_voice(self, role_id, voice_record, voice_path):
            calls.append(("ensure_voice", role_id))
            return voice_path

        def generate_sentence_batches(self, jobs):
            calls.append(("generate_sentence_batches", len(jobs)))
            yield []

        def generate_sentences(self, jobs):
            calls.append(("generate_sentences", len(jobs)))
            return []

        def close(self):
            calls.append(("close",))

    def build_adapter(config):
        calls.append(("build_adapter", config.tts_backend))
        return DummyAdapter()

    monkeypatch.setattr(controller_module, "_build_qwen_adapter", build_adapter)
    config = PipelineConfig(book_root=str(tmp_path / "book"))

    pipeline = controller_module._default_pipeline_factory(config, needs_llm=False, fake_tts=False)

    assert calls == []
    assert pipeline.tts_adapter.generate_sentences([]) == []
    assert calls == [("build_adapter", "native"), ("generate_sentences", 0)]
    pipeline.tts_adapter.close()
    assert calls == [("build_adapter", "native"), ("generate_sentences", 0), ("close",)]


def test_controller_saves_read_along_time_buffer_settings(tmp_path):
    paths = BookPaths(tmp_path / "book")
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)

    controller.save_read_along_settings(
        {
            "playback_speed": "1.25",
            "generation_mode": "fast",
            "buffer_limit": "2",
            "target_buffer_seconds": "12.5",
            "start_buffer_seconds": "4",
            "max_buffer_seconds": "20",
            "chapter_end_behavior": "continue",
        }
    )

    settings = controller.read_along_settings()
    assert settings["playback_speed"] == 1.25
    assert settings["generation_mode"] == "fast"
    assert settings["buffer_limit"] == 2
    assert settings["target_buffer_seconds"] == 12.5
    assert settings["max_buffer_seconds"] == 20.0
    assert settings["chapter_end_behavior"] == "continue"
    assert "start_buffer_seconds" not in settings
    assert "narrator_voice_type" not in settings


def test_controller_uses_target_buffer_seconds_as_initial_session_buffer(tmp_path):
    paths = BookPaths(tmp_path / "book")
    paths.chapter_text("chapter_001").parent.mkdir(parents=True)
    paths.chapter_text("chapter_001").write_text('Leigh said, "Right."', encoding="utf-8")
    _write_callie_registry(paths)
    registry = read_json(paths.registry)
    registry["narrator"]["voice_profile"] = {
        "description": "adult male narrator",
        "qwen_instruct": "adult male narrator",
    }
    registry["characters"]["leigh_adult"] = {
        "role_id": "leigh_adult",
        "profile_id": "leigh_adult",
        "person_id": "leigh",
        "display_name": "Leigh",
        "age_stage": "adult",
        "aliases": [],
        "identity_profile": {"age_stage": "adult", "gender": "female", "personality": []},
        "voice_identity": {"seed": 3, "differentiators": []},
        "voice_profile": {"description": "adult female", "qwen_instruct": "adult female"},
    }
    write_json_atomic(paths.registry, registry)
    controller = PrototypeUiController(book_root=paths.root, fake_tts=True)
    paths.annotation("chapter_001").parent.mkdir(parents=True)
    write_json_atomic(
        paths.annotation("chapter_001"),
        {"schema": "quote_attribution_v1", "roles": ["leigh_adult"], "quotes": [[1, 0]]},
    )
    controller.prepare_read_along_voices()
    settings = {
        "playback_speed": 1.0,
        "generation_mode": "balanced",
        "buffer_limit": 2,
        "target_buffer_seconds": 12.5,
        "max_buffer_seconds": 25.0,
    }

    session = controller.create_read_along_session(
        "chapter_001",
        controller.read_along_units("chapter_001"),
        settings,
    )

    assert session.start_buffer_seconds == 12.5
    assert session.target_buffer_seconds == 12.5
    session.end()
