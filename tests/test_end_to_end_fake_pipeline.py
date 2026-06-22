import shutil

from ebook_tts_pipeline.annotation.service import AnnotationService
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.ingestion import ChapterSplitter
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.pipeline import AudiobookPipeline
from ebook_tts_pipeline.tts.fake import FakeTtsAdapter


class TinyBookLlm:
    def complete_json(self, system_prompt, user_prompt):
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
            "script": [[0, 0, 0], [1, 1, 1], [0, 0, 2]],
        }


def test_fake_pipeline_from_whole_book_to_audio_outputs(tmp_path):
    book_root = tmp_path / "tiny_book"
    shutil.copytree("tests/fixtures/tiny_book", book_root)
    paths = BookPaths(book_root)

    split = ChapterSplitter().split_source_book(paths)
    assert split.chapters == ["chapter_001", "chapter_002"]

    pipeline = AudiobookPipeline(
        config=PipelineConfig(book_root=str(book_root), anthropic_api_key="fake"),
        annotation_service=AnnotationService(TinyBookLlm(), repair_retries=1),
        tts_adapter=FakeTtsAdapter(sample_rate=1000, samples_per_character=5),
        tokenizer=lambda text: ["It rained on the old road.", '"Hello," Elena said.'],
    )

    timeline = pipeline.run_chapter("chapter_001", book_title="Tiny Book", book_slug="tiny_book")

    assert timeline["sentences"][0]["sentence_idx"] == 0
    assert (book_root / "registry.json").exists()
    assert (book_root / "voices" / "_temp" / "chapter_001" / "tmp_001_default.qvp").exists()
    assert (book_root / "audio" / "chapter_001.wav").exists()
