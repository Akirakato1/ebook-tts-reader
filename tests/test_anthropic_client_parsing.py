from ebook_tts_pipeline.annotation.anthropic_client import parse_json_response_text
from ebook_tts_pipeline.annotation.prompts import render_annotation_prompt
from ebook_tts_pipeline.domain import Sentence


def test_parse_json_response_text_accepts_markdown_json_fence():
    payload = parse_json_response_text(
        '```json\n{"new_characters":[],"roles":["Narrator"],"types":["narration","dialogue","thought"],"script":[]}\n```'
    )

    assert payload["roles"] == ["Narrator"]


def test_annotation_prompt_explicitly_forbids_markdown_fences_and_names_narrator():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(0, "The hallway was dark.")],
        {"characters": {}},
    )

    assert "Do not wrap the JSON in Markdown code fences." in prompt
    assert "Use exactly \"Narrator\"" in prompt
