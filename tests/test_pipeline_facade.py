from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


class FakeLlmClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        return {
            "new_characters": [
                {
                    "name": "Elena",
                    "profile": {"age_range": "young adult"},
                    "voice": {
                        "description": "young woman, soft",
                        "qwen_instruct": "A soft young adult female voice.",
                    },
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
    assert (book_root / "audio" / "chapter_001.wav").exists()
    assert (book_root / "audio" / "chapter_001.timeline.json").exists()
    assert (book_root / "voices" / "elena.qvp").exists()


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
                        "profile": {"age_range": "young adult"},
                        "voice": {
                            "description": "young woman, soft",
                            "qwen_instruct": "A soft young adult female voice.",
                        },
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
    assert registry["characters"]["elena"]["display_name"] == "Elena"
