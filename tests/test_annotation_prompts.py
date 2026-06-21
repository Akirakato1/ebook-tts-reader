from ebook_tts_pipeline.annotation.prompts import render_annotation_prompt
from ebook_tts_pipeline.domain import Sentence


def test_annotation_prompt_requires_profile_object_for_new_characters():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Hello.")],
        {"characters": {}},
    )

    assert "profile must be a JSON object, never null, never a string" in prompt


def test_annotation_prompt_requires_one_script_row_per_sentence():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text='"Hello." "Hi." She waved.')],
        {"characters": {}},
    )

    assert "Never emit multiple script rows for the same sentence_idx" in prompt
    assert "choose the first or primary speaker" in prompt


def test_annotation_prompt_rejects_numbered_person_ids():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Callie waited.")],
        {"characters": {}},
    )

    assert "Do not append chapter, window, or sentence numbers to person_id or profile_id" in prompt


def test_annotation_prompt_uses_lean_character_profile_schema():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Callie waited.")],
        {"characters": {}},
    )

    assert "occupation" in prompt
    assert "person_id, age," not in prompt
    assert "timeline" not in prompt
    assert "same_person_as" not in prompt
    assert "narrative_notes" not in prompt


def test_annotation_prompt_uses_compact_character_summaries_not_voice_registry():
    registry = {
        "characters": {
            "callie_teen": {
                "role_id": "callie_teen",
                "profile_id": "callie_teen",
                "person_id": "callie",
                "display_name": "Callie",
                "age_stage": "teen",
                "aliases": ["Callie teen"],
                "identity_profile": {
                    "age_stage": "teen",
                    "gender": "female",
                    "personality": ["guarded", "timid", "wary", "quiet", "fragile", "extra"],
                    "race_or_ethnicity": None,
                    "accent": None,
                    "occupation": "student",
                },
                "voice_identity": {"seed": 123, "differentiators": ["darker timbre"]},
                "voice_variants": {
                    "default": {
                        "role_id": "callie_teen_default",
                        "display_name": "Callie_default",
                        "voice_profile": {
                            "description": "teen female; guarded",
                            "qwen_instruct": "A teen female voice.",
                        },
                        "voice_config_path": "voices/callie_teen_default.qvp",
                        "voice_config_hash": "abc",
                    }
                },
            }
        }
    }

    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Callie waited.")],
        registry,
        lock_registry=True,
    )

    assert '"name":"Callie"' in prompt
    assert '"aliases":["Callie teen"]' in prompt
    assert '"age_stage":"teen"' in prompt
    assert '"occupation":"student"' in prompt
    assert '"personality_type":"guarded, timid, wary, quiet, fragile"' in prompt
    assert "voice_variants" not in prompt
    assert "qwen_instruct" not in prompt
    assert "voice_config_path" not in prompt
    assert "voice_config_hash" not in prompt
    assert '"seed":' not in prompt
    assert '"role_id":' not in prompt
    assert '"person_id":' not in prompt


def test_locked_annotation_prompt_uses_proposed_new_characters():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(0, '"Hello," Akari said.')],
        {"characters": {"akari_adult": {"display_name": "Akari", "aliases": []}}},
        lock_registry=True,
    )

    assert "proposed_new_characters" in prompt
    assert "Do not add to new_characters" in prompt
    assert "new_characters: []" in prompt
