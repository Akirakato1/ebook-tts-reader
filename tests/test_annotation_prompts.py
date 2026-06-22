from ebook_tts_pipeline.annotation.prompts import render_annotation_prompt
from ebook_tts_pipeline.domain import Sentence, SentenceUnit


def test_annotation_prompt_requires_profile_object_for_local_speakers():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Hello.")],
        {"characters": {}},
    )

    assert "local_speakers" in prompt
    assert "profile must be a JSON object, never null, never a string" in prompt
    assert "new_characters" not in prompt


def test_annotation_prompt_requires_one_script_row_per_annotation_unit():
    prompt = render_annotation_prompt(
        "chapter_001",
        [
            SentenceUnit(idx=0, sentence_idx=0, text='"Hello."'),
            SentenceUnit(idx=1, sentence_idx=0, text="She waved."),
        ],
        {"characters": {}},
    )

    assert "script: list of [role_idx, type_idx, unit_idx]" in prompt
    assert "Allowed unit_idx values: [0, 1]" in prompt
    assert "script must contain exactly 2 rows, one for each allowed unit_idx" in prompt
    assert "Each annotation unit contains at most one non-narrator speaker section" in prompt
    assert "choose the first or primary speaker" not in prompt


def test_annotation_prompt_describes_role_allocation_units():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text='"Hello." Alice smiled.')],
        {"characters": {}},
        lock_registry=True,
    )

    assert "Each annotation unit contains at most one non-narrator speaker section." in prompt
    assert "If a unit contains quoted speech plus narrator context" in prompt
    assert "These units are for speaker labeling, not final TTS text." in prompt
    assert "deterministic script generation will extract narrator context later" in prompt
    assert "Do not add extra Narrator rows for said-tags or action beats inside a speaker-labeled unit." in prompt
    assert "Do not split or merge unit_idx values in your output." in prompt


def test_annotation_prompt_does_not_request_global_character_creation():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(idx=0, text="Callie waited.")],
        {"characters": {}},
    )

    assert "new_characters" not in prompt
    assert "proposed_new_characters" not in prompt
    assert "local_speakers" in prompt


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


def test_locked_annotation_prompt_uses_local_speakers_for_unregistered_disposable_roles():
    prompt = render_annotation_prompt(
        "chapter_001",
        [Sentence(0, '"Hello," Akari said.')],
        {"characters": {"akari_adult": {"display_name": "Akari", "aliases": []}}},
        lock_registry=True,
    )

    assert "local_speakers" in prompt
    assert "chapter-scoped temporary speakers" in prompt
    assert "unnamed one-off" in prompt
    assert "No approval is required for local_speakers" in prompt
    assert "proposed_new_characters" not in prompt
    assert "new_characters" not in prompt
