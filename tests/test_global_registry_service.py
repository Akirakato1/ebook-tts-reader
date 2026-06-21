from ebook_tts_pipeline.annotation.global_registry import (
    GlobalRegistryChapter,
    GlobalRegistryService,
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
    assert "Akari Nakayama waved" in prompt


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
