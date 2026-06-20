from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager


def test_registry_adds_new_character_with_stable_voice_identity(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")

    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Elena",
                "profile": {"age_range": "young adult", "gender": "female"},
                "voice": {
                    "description": "young woman, soft",
                    "qwen_instruct": "A soft young adult female voice.",
                },
            }
        ],
    )

    registry = read_json(paths.registry)
    elena = registry["characters"]["elena"]
    assert elena["display_name"] == "Elena"
    assert elena["first_seen"] == "chapter_001"
    assert elena["voice_config_path"] is None
    assert isinstance(elena["voice_identity"]["seed"], int)
    assert elena["voice_identity"]["differentiators"]


def test_similar_character_receives_different_voice_differentiator(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    repeated = {
        "profile": {"age_range": "young adult", "gender": "female"},
        "voice": {
            "description": "young woman, soft",
            "qwen_instruct": "A soft young adult female voice.",
        },
    }

    manager.add_new_characters(chapter="chapter_001", new_characters=[{"name": "Elena", **repeated}])
    manager.add_new_characters(chapter="chapter_002", new_characters=[{"name": "Mira", **repeated}])

    registry = read_json(paths.registry)
    elena_voice = registry["characters"]["elena"]["voice_profile"]["qwen_instruct"]
    mira_voice = registry["characters"]["mira"]["voice_profile"]["qwen_instruct"]
    assert elena_voice != mira_voice


def test_registry_rejects_alias_collision(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {"display_name": "Narrator"},
            "characters": {
                "elena": {
                    "display_name": "Elena",
                    "aliases": ["Lena"],
                    "voice_profile": {"description": "soft", "qwen_instruct": "soft"},
                }
            },
        },
    )
    manager = RegistryManager(paths)

    try:
        manager.add_new_characters(
            chapter="chapter_002",
            new_characters=[
                {
                    "name": "Lena",
                    "profile": {},
                    "voice": {"description": "x", "qwen_instruct": "x"},
                }
            ],
        )
    except ValueError as exc:
        assert "collides with existing character or alias" in str(exc)
    else:
        raise AssertionError("Expected alias collision")
