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
