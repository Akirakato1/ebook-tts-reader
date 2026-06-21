from ebook_tts_pipeline.annotation.global_registry import (
    GlobalRegistryChapter,
    GlobalRegistryService,
    compact_registry_for_global_prompt,
    render_global_registry_prompt,
)


class RecordingGlobalClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self.payload


def test_render_global_registry_prompt_requests_canonical_characters_only():
    prompt = render_global_registry_prompt(
        book_title="Demo Book",
        registry={"characters": {}},
        chapters=[
            GlobalRegistryChapter(
                chapter="chapter_001",
                title="Chapter One",
                text="Akari Nakayama waved. Akari smiled.",
            )
        ],
    )

    assert "Demo Book" in prompt
    assert "Chapter One" in prompt
    assert "canonical character registry" in prompt
    assert "Do not produce sentence-level annotation" in prompt
    assert "Existing registry is authoritative" in prompt
    assert "Do not recreate" in prompt
    assert "Return new characters and existing-character updates" in prompt
    assert "Do not echo unchanged registry records" in prompt
    assert "Akari Nakayama waved" in prompt


def test_global_registry_prompt_uses_minimal_character_summaries():
    registry = {
        "characters": {
            "akari_adult": {
                "role_id": "akari_adult",
                "profile_id": "akari_adult",
                "person_id": "akari",
                "display_name": "Akari Nakayama",
                "age": 31,
                "age_stage": "adult",
                "aliases": ["Akari", "Ms. Nakayama"],
                "same_person_as": ["Akari child"],
                "identity_profile": {
                    "age": 31,
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["careful", "warm", "tired", "guarded", "precise", "extra"],
                    "race_or_ethnicity": "Japanese",
                    "accent": "Tokyo",
                    "occupation": "barista",
                },
                "character_profile": {"gender": "female"},
                "narrative_notes": "Long backstory that is useful to a human editor but not the prompt.",
                "voice_identity": {"seed": 123, "differentiators": ["brighter timbre"]},
                "voice_variants": {
                    "default": {
                        "voice_profile": {
                            "description": "adult female",
                            "qwen_instruct": "A generated Qwen voice instruction.",
                        },
                        "voice_config_hash": "abc",
                    }
                },
                "global_evidence": ["chapter evidence should not be resent"],
            }
        }
    }

    compact = compact_registry_for_global_prompt(registry)
    prompt = render_global_registry_prompt(
        book_title="Demo Book",
        registry=registry,
        chapters=[GlobalRegistryChapter(chapter="chapter_002", title="Next", text="Akari returned.")],
    )

    assert compact == [
        {
            "name": "Akari Nakayama",
            "age_stage": "adult",
            "gender": "female",
            "race_or_accent": "Japanese; Tokyo accent",
            "occupation": "barista",
            "personality_type": "careful, warm, tired, guarded, precise",
        }
    ]
    assert "Existing character summaries" in prompt
    assert "Akari Nakayama" in prompt
    assert "qwen_instruct" not in prompt
    assert '"role_id":' not in prompt
    assert '"profile_id":' not in prompt
    assert '"person_id":' not in prompt
    assert '"aliases":' not in prompt
    assert '"same_person_as":' not in prompt
    assert "voice_variants" not in prompt
    assert "voice_config_hash" not in prompt
    assert "global_evidence" not in prompt
    assert "narrative_notes" not in prompt


def test_global_registry_service_returns_characters_with_evidence():
    client = RecordingGlobalClient(
        {
            "characters": [
                {
                    "name": "Akari Nakayama",
                    "profile": {
                        "profile_id": "akari_nakayama_adult",
                        "person_id": "akari_nakayama",
                        "age_stage": "adult",
                        "gender": "female",
                        "personality": ["professional", "direct"],
                        "aliases": ["Akari"],
                    },
                    "evidence": [
                        {"chapter": "chapter_001", "note": "Introduced by full name"},
                    ],
                }
            ]
        }
    )
    service = GlobalRegistryService(client=client)

    result = service.discover_characters(
        book_title="Demo Book",
        registry={"characters": {}},
        chapters=[
            GlobalRegistryChapter(
                chapter="chapter_001",
                title="Chapter One",
                text="Akari Nakayama waved.",
            )
        ],
    )

    assert len(client.calls) == 1
    assert result.characters[0]["name"] == "Akari Nakayama"
    assert result.characters[0]["profile"]["aliases"] == ["Akari"]
    assert result.characters[0]["evidence"][0]["chapter"] == "chapter_001"
