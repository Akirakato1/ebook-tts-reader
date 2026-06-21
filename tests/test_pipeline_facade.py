import re

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.domain import AnnotationResult
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
            "new_characters": [
                {
                    "name": "Elena",
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
    assert (book_root / "voices" / "elena_adult_default.qvp").exists()


def test_pipeline_annotates_multi_window_chapter_and_preserves_new_characters(tmp_path):
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
                "new_characters": [],
                "roles": ["Narrator", "Elena"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[1, 2, 2]],
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
    assert [character["name"] for character in annotation.new_characters] == ["Elena"]
    assert annotation.script == [(0, 0, 0), (1, 1, 1), (1, 2, 2)]
    assert registry["characters"]["elena_adult"]["display_name"] == "Elena"


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
    assert registry["characters"]["akari_nakayama_adult"]["global_evidence"][0]["chapter"] == "chapter_001"


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
                "proposed_new_characters": [
                    {
                        "name": "Mystery",
                        "profile": {"age_stage": "adult", "gender": "unknown", "personality": ["quiet"]},
                    }
                ],
                "roles": ["Mystery"],
                "types": ["narration", "dialogue", "thought"],
                "script": [[0, 1, 0]],
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
    assert annotation.proposed_new_characters[0]["name"] == "Mystery"


def test_pipeline_prepares_default_and_internal_voice_variants_with_cache_invalidation_and_force(tmp_path):
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

    assert [call["role_id"] for call in adapter.calls] == ["elena_adult_default", "elena_adult_internal"]
    registry = pipeline.registry.load()
    variants = registry["characters"]["elena_adult"]["voice_variants"]
    assert variants["default"]["voice_config_hash"]
    assert variants["internal"]["voice_config_hash"]

    variants["internal"]["voice_profile"]["qwen_instruct"] += " More inward."
    pipeline.registry.save(registry)
    pipeline.prepare_voices_for_annotation(annotation)

    assert [call["role_id"] for call in adapter.calls] == [
        "elena_adult_default",
        "elena_adult_internal",
        "elena_adult_internal",
    ]
    assert adapter.calls[-1]["force"] is True

    pipeline.prepare_voices_for_annotation(annotation, force_regenerate=True)

    assert [call["role_id"] for call in adapter.calls] == [
        "elena_adult_default",
        "elena_adult_internal",
        "elena_adult_internal",
        "elena_adult_default",
        "elena_adult_internal",
    ]
    assert [call["force"] for call in adapter.calls[-2:]] == [True, True]


def test_pipeline_synthesizes_tts_windows_separately(tmp_path):
    adapter = WindowRecordingTtsAdapter()
    pipeline = AudiobookPipeline(
        config=PipelineConfig(
            book_root=str(tmp_path / "demo"),
            anthropic_api_key="fake",
            max_tts_window_chars=10,
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
