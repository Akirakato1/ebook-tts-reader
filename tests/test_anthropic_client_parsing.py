import pytest

from ebook_tts_pipeline.annotation.anthropic_client import (
    AnnotationModelOutputError,
    parse_json_response_text,
)
from ebook_tts_pipeline.annotation.prompts import render_annotation_prompt
from ebook_tts_pipeline.domain import Sentence


def test_parse_json_response_text_accepts_markdown_json_fence():
    payload = parse_json_response_text(
        '```json\n{"new_characters":[],"roles":["Narrator"],"types":["narration","dialogue","thought"],"script":[]}\n```'
    )

    assert payload["roles"] == ["Narrator"]


def test_parse_json_response_text_rejects_empty_response_with_clear_error():
    with pytest.raises(AnnotationModelOutputError, match="empty"):
        parse_json_response_text("   ")


def test_parse_json_response_text_includes_preview_for_non_json_response():
    with pytest.raises(AnnotationModelOutputError, match="I cannot"):
        parse_json_response_text("I cannot return that as JSON.")


def test_annotation_prompt_explicitly_forbids_markdown_fences_and_names_narrator():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(0, "The hallway was dark.")],
        {"characters": {}},
    )

    assert "Do not wrap the JSON in Markdown code fences." in prompt
    assert "Use exactly \"Narrator\"" in prompt
    assert "Do not include Narrator in new_characters." in prompt
    assert "profile required fields: age_stage, gender, personality." in prompt
    assert "Allowed unit_idx values: [0]" in prompt
    assert "script must contain exactly 1 rows" in prompt
