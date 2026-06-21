from pathlib import Path

import pytest

from ebook_tts_pipeline.epub_ingestion import EpubExtractResult
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
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

    def initialize_if_missing(self, book_title, book_slug):
        self.calls.append(("initialize", book_title, book_slug))
        if not self.paths.registry.exists():
            write_json_atomic(
                self.paths.registry,
                {"book": {"title": book_title, "slug": book_slug}, "characters": {}},
            )

    def load(self):
        return read_json(self.paths.registry)


class FakePipeline:
    def __init__(self, config, calls):
        self.paths = BookPaths(config.book_root)
        self.calls = calls
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

    def prepare_voices_for_annotation(self, annotation):
        self.calls.append(("prepare_voices",))

    def synthesize_chapter_from_tts_script(self, chapter):
        self.calls.append(("synthesize", chapter))
        self.paths.chapter_audio(chapter).parent.mkdir(parents=True, exist_ok=True)
        self.paths.chapter_audio(chapter).write_bytes(b"wav")


def fake_pipeline_factory(calls):
    def factory(config, needs_llm, fake_tts):
        calls.append(("factory", needs_llm, fake_tts))
        return FakePipeline(config, calls)

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
    assert "teen female" in character["voice_variants"]["default"]["voice_profile"]["qwen_instruct"]
    assert character["voice_variants"]["default"].get("voice_config_hash") is None


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
    assert ("prepare_voices",) in calls
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
