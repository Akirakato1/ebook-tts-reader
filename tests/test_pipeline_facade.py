import re

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionService
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult, Sentence, SentenceArtifact
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter
import numpy as np


class FakeLlmClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        return {
            "local_speakers": [
                {
                    "local_id": "tmp_001",
                    "label": "Elena",
                    "profile": {"age_stage": "adult", "gender": "female", "personality": ["soft"]},
                }
            ],
            "roles": ["Narrator", "Elena"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, 0], [1, 1, 1]],
        }


class QueuedLlmClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        return self.payloads.pop(0)


class SplitSensitiveLlmClient:
    def __init__(self, max_sentences):
        self.max_sentences = max_sentences
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        sentence_ids = [int(match) for match in re.findall(r"^\[(\d+)\]", user_prompt, flags=re.MULTILINE)]
        self.calls.append(sentence_ids)
        if len(sentence_ids) > self.max_sentences:
            raise AnnotationModelOutputError("Anthropic returned non-JSON content")
        return {
            "new_characters": [],
            "roles": ["Narrator"],
            "types": ["narration", "dialogue", "thought"],
            "script": [[0, 0, sentence_id] for sentence_id in sentence_ids],
        }


class CountingTtsAdapter(FakeTtsAdapter):
    def __init__(self):
        super().__init__(sample_rate=1000, samples_per_character=5)
        self.calls = []
        self.role_voice_paths = {}

    def ensure_voice(self, role_id, voice_record, voice_path):
        self.calls.append(
            {
                "role_id": role_id,
                "force": bool(voice_record.get("_force_regenerate")),
                "path": str(voice_path),
            }
        )
        return super().ensure_voice(role_id, voice_record, voice_path)


class WindowRecordingTtsAdapter:
    def __init__(self):
        self.calls = []

    def ensure_voice(self, role_id, voice_record, voice_path):
        return voice_path

    def generate_sentences(self, jobs):
        self.calls.append([job["sentence_idx"] for job in jobs])
        return [
            GeneratedSentenceAudio(
                sentence_idx=int(job["sentence_idx"]),
                role=str(job["role"]),
                speech_type=str(job["type"]),
                samples=np.ones(10, dtype=np.float32),
                sample_rate=1000,
            )
            for job in jobs
        ]


class FakeGlobalRegistryService:
    def __init__(self, characters):
        self.characters = characters
        self.calls = []

    def discover_characters(self, book_title, registry, chapters):
        self.calls.append(
            {
                "book_title": book_title,
                "chapters": [chapter.chapter for chapter in chapters],
                "registry": registry,
            }
        )
        return type("GlobalResult", (), {"characters": self.characters})()


class QueuedGlobalRegistryService:
    def __init__(self, character_batches):
        self.character_batches = list(character_batches)
        self.calls = []

    def discover_characters(self, book_title, registry, chapters):
        self.calls.append(
            {
                "book_title": book_title,
                "chapters": [chapter.chapter for chapter in chapters],
                "registry": registry,
            }
        )
        return type("GlobalResult", (), {"characters": self.character_batches.pop(0)})()


def test_pipeline_runs_tiny_chapter_with_fake_adapters(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        'It rained. "Hello," Elena said.',
        encoding="utf-8",
    )

    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(FakeLlmClient(), repair_retries=1),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["It rained.", '"Hello," Elena said.'],
    )

    result = pipeline.run_chapter("chapter_001", book_title="Demo", book_slug="demo")

    assert result["chapter"] == "chapter_001"
    assert (book_root / "sentence_segments" / "chapter_001.sentences.json").exists()
    assert (book_root / "annotations" / "chapter_001.annotation.json").exists()
    assert (book_root / "tts_scripts" / "chapter_001.tts_script.json").exists()
    assert (book_root / "tts_scripts" / "chapter_001.qwen_script.txt").exists()
    assert (book_root / "audio" / "chapter_001.wav").exists()
    assert (book_root / "audio" / "chapter_001.timeline.json").exists()
    assert pipeline.registry.load()["characters"] == {}
    assert (book_root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp").exists()


def test_pipeline_annotates_multi_window_chapter_converts_legacy_new_characters_to_local_speakers(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        'It rained. "Hello," Elena said. Later she left.',
        encoding="utf-8",
    )
    client = QueuedLlmClient(
        [
            {
                "new_characters": [
                    {
                        "name": "Elena",
                        "profile": {"age_stage": "adult", "gender": "female", "personality": ["soft"]},
                    }
                ],
                "roles": ["Narrator", "Elena"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 0], [1, 1, 1]],
            },
            {
                "local_speakers": [
                    {
                        "local_id": "tmp_001",
                        "label": "Elena",
                        "profile": {"age_stage": "adult", "gender": "female", "personality": ["soft"]},
                    }
                ],
                "roles": ["Narrator", "Elena"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 0, 2]],
            },
        ]
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(book_root),
            anthropic_api_key="fake",
            max_llm_window_chars=31,
        ),
        annotation_service=AnnotationService(client, repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: [
            "It rained.",
            '"Hello," Elena said.',
            "Later she left.",
        ],
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.segment_chapter("chapter_001")
    annotation = pipeline.annotate_chapter("chapter_001")
    registry = pipeline.registry.load()

    assert client.calls == 2
    assert annotation.new_characters == []
    assert [speaker["label"] for speaker in annotation.local_speakers] == ["Elena"]
    assert annotation.script == [(0, 0, 0), (1, 1, 1), (0, 0, 2)]
    assert registry["characters"] == {}


def test_pipeline_splits_annotation_window_after_unparseable_model_output(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        "One. Two. Three. Four.",
        encoding="utf-8",
    )
    client = SplitSensitiveLlmClient(max_sentences=2)
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(book_root),
            anthropic_api_key="fake",
            max_llm_window_chars=1000,
        ),
        annotation_service=AnnotationService(client, repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["One.", "Two.", "Three.", "Four."],
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.segment_chapter("chapter_001")
    annotation = pipeline.annotate_chapter("chapter_001")

    assert client.calls == [[0, 1, 2, 3], [0, 1], [2, 3]]
    assert annotation.script == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]


def test_pipeline_caps_initial_annotation_windows_by_sentence_count(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        "One. Two. Three. Four.",
        encoding="utf-8",
    )
    client = SplitSensitiveLlmClient(max_sentences=2)
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(book_root),
            anthropic_api_key="fake",
            max_llm_window_chars=1000,
            max_llm_window_sentences=2,
        ),
        annotation_service=AnnotationService(client, repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["One.", "Two.", "Three.", "Four."],
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.segment_chapter("chapter_001")
    annotation = pipeline.annotate_chapter("chapter_001")

    assert client.calls == [[0, 1], [2, 3]]
    assert annotation.script == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]


def test_pipeline_builds_global_registry_from_segmented_chapters(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text("Akari Nakayama waved.", encoding="utf-8")
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["Akari Nakayama waved."],
        global_registry_service=FakeGlobalRegistryService(
            [
                {
                    "name": "Akari Nakayama",
                    "profile": {
                        "profile_id": "akari_nakayama_adult",
                        "person_id": "akari_nakayama",
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["professional"],
                        "aliases": ["Akari"],
                    },
                    "evidence": [{"chapter": "chapter_001", "note": "Full name"}],
                }
            ]
        ),
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.segment_chapter("chapter_001")
    count = pipeline.build_global_registry(book_title="Demo")

    registry = pipeline.registry.load()
    assert count == 1
    assert registry["characters"]["akari_nakayama_adult"]["display_name"] == "Akari Nakayama"
    assert "global_evidence" not in registry["characters"]["akari_nakayama_adult"]


def test_pipeline_builds_global_registry_initializes_missing_registry(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text("Akari waved.", encoding="utf-8")
    service = FakeGlobalRegistryService([])
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["Akari waved."],
        global_registry_service=service,
    )

    count = pipeline.build_global_registry(book_title="Demo", book_slug="demo")

    registry = pipeline.registry.load()
    assert count == 0
    assert registry["book"] == {"title": "Demo", "slug": "demo"}
    assert service.calls[0]["registry"]["characters"] == {}


def test_pipeline_builds_global_registry_in_chapter_chunks_with_updated_registry(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text("Akari waved.", encoding="utf-8")
    (chapter_dir / "chapter_002.txt").write_text("She smiled.", encoding="utf-8")
    (chapter_dir / "chapter_003.txt").write_text("Bento arrived.", encoding="utf-8")
    service = QueuedGlobalRegistryService(
        [
            [
                {
                    "name": "Akari",
                    "profile": {
                        "profile_id": "akari_adult",
                        "person_id": "akari",
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["careful"],
                    },
                    "evidence": [{"chapter": "chapter_001", "note": "Introduced"}],
                }
            ],
            [
                {
                    "name": "Bento",
                    "profile": {
                        "profile_id": "bento_adult",
                        "person_id": "bento",
                        "age_stage": "adult",
                        "gender": "male",
                        "personality": ["warm"],
                    },
                    "evidence": [{"chapter": "chapter_003", "note": "Introduced"}],
                }
            ],
        ]
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(book_root),
            anthropic_api_key="fake",
            global_registry_window_chars=25,
        ),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: [text],
        global_registry_service=service,
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    count = pipeline.build_global_registry(book_title="Demo")

    registry = pipeline.registry.load()
    assert count == 2
    assert [call["chapters"] for call in service.calls] == [
        ["chapter_001", "chapter_002"],
        ["chapter_003"],
    ]
    assert service.calls[0]["registry"]["characters"] == {}
    assert "akari_adult" in service.calls[1]["registry"]["characters"]
    assert registry["characters"]["akari_adult"]["display_name"] == "Akari"
    assert registry["characters"]["bento_adult"]["display_name"] == "Bento"


def test_pipeline_locked_annotation_does_not_mutate_registry_with_new_characters(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text('"Hello," Mystery said.', encoding="utf-8")
    client = QueuedLlmClient(
        [
            {
                "new_characters": [
                    {
                        "name": "Mystery",
                        "profile": {"age_stage": "adult", "gender": "unknown", "personality": ["quiet"]},
                    }
                ],
                "local_speakers": [
                    {
                        "local_id": "tmp_001",
                        "label": "Mystery",
                        "profile": {"age_stage": "adult", "gender": "unknown", "personality": ["quiet"]},
                    }
                ],
                "roles": ["Narrator", "Mystery"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[1, 1, 0]],
            }
        ]
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(client, repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ['"Hello," Mystery said.'],
    )

    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.segment_chapter("chapter_001")
    annotation = pipeline.annotate_chapter("chapter_001", lock_registry=True)

    assert pipeline.registry.load()["characters"] == {}
    assert annotation.new_characters == []
    assert annotation.local_speakers[0]["label"] == "Mystery"


def test_pipeline_builds_temp_registry_and_prepares_local_speaker_voices(tmp_path):
    adapter = CountingTtsAdapter()
    book_root = tmp_path / "demo"
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=adapter,
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    artifact = SentenceArtifact(
        chapter="chapter_013",
        source_path="chapters/chapter_013.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, '"Move along," the guard said.')],
    )
    write_json_atomic(pipeline.paths.sentence_artifact("chapter_013"), artifact.to_dict())
    annotation = AnnotationResult(
        new_characters=[],
        local_speakers=[
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
        roles=["tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0)],
    )

    jobs = pipeline.build_sentence_jobs("chapter_013", annotation)
    pipeline.prepare_voices_for_annotation(annotation, chapter="chapter_013")

    registry = pipeline.registry.load()
    temp_registry = read_json(pipeline.paths.chapter_temp_registry("chapter_013"))
    assert registry["characters"] == {}
    assert temp_registry["speakers"]["tmp_001"]["label"] == "Security Guard"
    assert jobs[0]["role_id"] == "chapter_013_tmp_001"
    assert jobs[0]["voice_config_path"] == "voices/_temp/chapter_013/tmp_001.qvp"
    assert (book_root / "voices" / "_temp" / "chapter_013" / "tmp_001.qvp").exists()
    assert adapter.calls == [
        {
            "role_id": "chapter_013_tmp_001",
            "force": False,
            "path": str(book_root / "voices" / "_temp" / "chapter_013" / "tmp_001.qvp"),
        }
    ]


def test_pipeline_quote_annotation_builds_single_voice_script_from_raw_chapter(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        'Callie said, "Stay here." She left.',
        encoding="utf-8",
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        quote_attribution_service=QuoteAttributionService(
            QueuedLlmClient(
                [
                    {
                        "roles": ["callie_child"],
                        "quotes": [[1, 0, "dialogue"]],
                    }
                ]
            )
        ),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["Callie said, ", '"Stay here."', " She left."],
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.registry.add_new_characters(
        chapter="global_registry",
        new_characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "child",
                    "gender": "female",
                    "personality": ["guarded"],
                },
            }
        ],
    )

    pipeline.segment_chapter("chapter_001")
    annotation = pipeline.annotate_chapter("chapter_001")
    jobs = pipeline.build_sentence_jobs("chapter_001", annotation)

    saved = read_json(pipeline.paths.annotation("chapter_001"))
    assert saved["schema"] == "quote_attribution_v1"
    assert saved["quotes"] == [[1, 0]]
    assert "script" not in saved
    assert "types" not in saved
    assert [job["role"] for job in jobs] == ["Narrator", "callie_child", "Narrator"]
    assert pipeline.paths.qwen_script("chapter_001").read_text(encoding="utf-8") == (
        "Narrator: Callie said,\n"
        'callie_child: "Stay here."\n'
        "Narrator: She left.\n"
    )


def test_pipeline_prepare_voices_refreshes_existing_quote_tts_script_paths(tmp_path):
    adapter = CountingTtsAdapter()
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text(
        'Callie said, "Stay here."',
        encoding="utf-8",
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        quote_attribution_service=QuoteAttributionService(
            QueuedLlmClient(
                [
                    {
                        "roles": ["callie_child"],
                        "quotes": [[1, 0, "dialogue"]],
                    }
                ]
            )
        ),
        tts_adapter=adapter,
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.registry.add_new_characters(
        chapter="global_registry",
        new_characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "child",
                    "gender": "female",
                    "personality": ["guarded"],
                },
            }
        ],
    )

    annotation = pipeline.annotate_chapter("chapter_001")
    pipeline.build_sentence_jobs("chapter_001", annotation)
    before = read_json(pipeline.paths.tts_script("chapter_001"))

    pipeline.prepare_voices_for_annotation(annotation, chapter="chapter_001")
    after = read_json(pipeline.paths.tts_script("chapter_001"))

    assert before["jobs"][1]["voice_config_path"] is None
    assert after["jobs"][1]["voice_config_path"] == "voices/callie_child.qvp"
    assert after["windows"][0]["jobs"][1]["voice_config_path"] == "voices/callie_child.qvp"
    assert "batches" not in after["windows"][0]
    assert after["windows"][0]["qwen_text"] == 'Narrator: Callie said,\ncallie_child: "Stay here."'


def test_pipeline_locked_annotation_accepts_unique_registry_display_names(tmp_path):
    book_root = tmp_path / "demo"
    chapter_dir = book_root / "chapters"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "chapter_001.txt").write_text('"Come here," Buddy Waleski said.', encoding="utf-8")
    client = QueuedLlmClient(
        [
            {
                "new_characters": [],
                "roles": ["Narrator", "Buddy Waleski"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[1, 1, 0]],
            }
        ]
    )
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(client, repair_retries=0),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ['"Come here," Buddy Waleski said.'],
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.registry.add_new_characters(
        chapter="global_registry",
        new_characters=[
            {
                "name": "Buddy Waleski",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["aggressive"],
                    "aliases": ["Buddy Waleski adult"],
                },
            }
        ],
    )
    pipeline.segment_chapter("chapter_001")

    annotation = pipeline.annotate_chapter("chapter_001", lock_registry=True)

    assert annotation.roles == ["Buddy Waleski"]


def test_pipeline_prepares_single_voice_with_cache_invalidation_and_force(tmp_path):
    adapter = CountingTtsAdapter()
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(tmp_path / "demo"), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=adapter,
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    pipeline.registry.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Elena",
                "profile": {"age_stage": "adult", "gender": "female", "personality": ["soft"]},
            }
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0), (0, 2, 1)],
    )

    pipeline.prepare_voices_for_annotation(annotation)
    pipeline.prepare_voices_for_annotation(annotation)

    assert [call["role_id"] for call in adapter.calls] == ["elena_adult"]
    registry = pipeline.registry.load()
    character = registry["characters"]["elena_adult"]
    assert character["voice_config_hash"]

    character["voice_profile"]["qwen_instruct"] += " More forceful."
    pipeline.registry.save(registry)
    pipeline.prepare_voices_for_annotation(annotation)

    assert [call["role_id"] for call in adapter.calls] == [
        "elena_adult",
        "elena_adult",
    ]
    assert adapter.calls[-1]["force"] is True

    pipeline.prepare_voices_for_annotation(annotation, force_regenerate=True)

    assert [call["role_id"] for call in adapter.calls] == [
        "elena_adult",
        "elena_adult",
        "elena_adult",
    ]
    assert adapter.calls[-1]["force"] is True


def test_pipeline_prepares_single_voice_for_local_speaker(tmp_path):
    adapter = CountingTtsAdapter()
    book_root = tmp_path / "demo"
    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=adapter,
    )
    pipeline.registry.initialize_if_missing(book_title="Demo", book_slug="demo")
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, '"Move along," the guard said.')],
    )
    write_json_atomic(pipeline.paths.sentence_artifact("chapter_001"), artifact.to_dict())
    annotation = AnnotationResult(
        new_characters=[],
        local_speakers=[
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
        roles=["tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0)],
    )

    jobs = pipeline.build_sentence_jobs("chapter_001", annotation)
    pipeline.prepare_voices_for_annotation(annotation, chapter="chapter_001")

    temp_registry = read_json(pipeline.paths.chapter_temp_registry("chapter_001"))
    speaker = temp_registry["speakers"]["tmp_001"]
    assert "voice_variants" not in speaker
    assert speaker["role_id"] == "chapter_001_tmp_001"
    assert speaker["voice_config_path"] == "voices/_temp/chapter_001/tmp_001.qvp"
    assert jobs[0]["role"] == "chapter_001_tmp_001"
    assert jobs[0]["role_id"] == "chapter_001_tmp_001"
    assert jobs[0]["voice_config_path"] == "voices/_temp/chapter_001/tmp_001.qvp"
    assert (book_root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp").exists()
    assert adapter.calls == [
        {
            "role_id": "chapter_001_tmp_001",
            "force": False,
            "path": str(book_root / "voices" / "_temp" / "chapter_001" / "tmp_001.qvp"),
        }
    ]


def test_pipeline_synthesizes_tts_windows_separately(tmp_path):
    adapter = WindowRecordingTtsAdapter()
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(tmp_path / "demo"),
            anthropic_api_key="fake",
            max_tts_window_chars=20,
            pause_between_sentences_ms=0,
        ),
        annotation_service=AnnotationService(QueuedLlmClient([]), repair_retries=0),
        tts_adapter=adapter,
    )

    pipeline.synthesize_jobs(
        "chapter_001",
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "123456"},
            {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "abcdef"},
        ],
    )

    assert adapter.calls == [[0], [1]]
