from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager, resolve_effective_voice


def test_registry_adds_new_character_with_stable_voice_identity(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")

    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Elena",
                "profile": {
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["soft", "hesitant"],
                },
            }
        ],
    )

    registry = read_json(paths.registry)
    elena = registry["characters"]["elena_adult"]
    assert elena["display_name"] == "Elena"
    assert elena["profile_id"] == "elena_adult"
    assert elena["person_id"] == "elena"
    assert elena["age_stage"] == "adult"
    assert elena["first_seen"] == "chapter_001"
    assert set(elena["voice_variants"]) == {"default", "internal"}
    assert elena["voice_variants"]["default"]["role_id"] == "elena_adult_default"
    assert elena["voice_variants"]["internal"]["role_id"] == "elena_adult_internal"
    assert elena["voice_variants"]["default"]["voice_config_path"] is None
    assert elena["voice_variants"]["internal"]["voice_config_path"] is None
    assert isinstance(elena["voice_identity"]["seed"], int)
    assert elena["voice_identity"]["differentiators"]
    voice_prompt = elena["voice_variants"]["default"]["voice_profile"]["qwen_instruct"]
    assert "adult female" in voice_prompt
    assert "soft, hesitant" in voice_prompt
    assert (
        elena["voice_variants"]["default"]["voice_profile"]["qwen_instruct"]
        != elena["voice_variants"]["internal"]["voice_profile"]["qwen_instruct"]
    )


def test_similar_character_receives_different_voice_differentiator(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    repeated = {
        "profile": {"age_stage": "adult", "gender": "female", "personality": ["soft"]},
    }

    manager.add_new_characters(chapter="chapter_001", new_characters=[{"name": "Elena", **repeated}])
    manager.add_new_characters(chapter="chapter_002", new_characters=[{"name": "Mira", **repeated}])

    registry = read_json(paths.registry)
    elena_voice = registry["characters"]["elena_adult"]["voice_variants"]["default"]["voice_profile"]["qwen_instruct"]
    mira_voice = registry["characters"]["mira_adult"]["voice_variants"]["default"]["voice_profile"]["qwen_instruct"]
    assert elena_voice != mira_voice


def test_registry_creates_distinct_age_stage_profiles_for_same_person(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")

    manager.add_new_characters(
        chapter="interlude_001",
        new_characters=[
            {
                "name": "Callie",
                "profile": {
                    "person_id": "callie",
                    "age": 14,
                    "age_stage": "teen",
                    "gender": "female",
                    "personality": ["guarded", "timid"],
                    "timeline": "interlude_past",
                    "narrative_notes": "Victim of grooming and exploitation; not a romance.",
                },
            },
            {
                "name": "Callie",
                "profile": {
                    "person_id": "callie",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["hardened", "protective"],
                    "timeline": "present",
                    "same_person_as": ["callie_teen"],
                },
            },
        ],
    )

    registry = read_json(paths.registry)
    assert set(registry["characters"]) == {"callie_teen", "callie_adult"}
    teen = registry["characters"]["callie_teen"]
    adult = registry["characters"]["callie_adult"]
    assert teen["person_id"] == adult["person_id"] == "callie"
    assert teen["age"] == 14
    assert teen["narrative_notes"] == "Victim of grooming and exploitation; not a romance."
    assert adult["same_person_as"] == ["callie_teen"]
    assert "grooming" not in teen["voice_variants"]["default"]["voice_profile"]["qwen_instruct"].lower()
    assert "callie_teen_default" == teen["voice_variants"]["default"]["role_id"]
    assert "callie_adult_default" == adult["voice_variants"]["default"]["role_id"]


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
                    "profile": {"age_stage": "adult", "gender": "female", "personality": ["quiet"]},
                }
            ],
        )
    except ValueError as exc:
        assert "collides with existing character or alias" in str(exc)
    else:
        raise AssertionError("Expected alias collision")


def test_global_registry_merge_adds_alias_to_existing_character(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Akari",
                "profile": {
                    "profile_id": "akari_nakayama_adult",
                    "person_id": "akari_nakayama",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["professional"],
                    "aliases": ["Akari"],
                },
            }
        ],
    )

    manager.merge_global_characters(
        chapter="global_registry",
        characters=[
            {
                "name": "Akari Nakayama",
                "profile": {
                    "profile_id": "akari_nakayama_adult",
                    "person_id": "akari_nakayama",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["professional", "direct"],
                    "aliases": ["Akari", "Miss Nakayama"],
                },
                "evidence": [{"chapter": "chapter_003", "note": "Full name appears"}],
            }
        ],
    )

    registry = read_json(paths.registry)
    assert list(registry["characters"]) == ["akari_nakayama_adult"]
    akari = registry["characters"]["akari_nakayama_adult"]
    assert akari["display_name"] == "Akari"
    assert "Miss Nakayama" in akari["aliases"]
    assert "direct" in akari["identity_profile"]["personality"]
    assert akari["global_evidence"][0]["chapter"] == "chapter_003"


def test_global_registry_merge_updates_key_character_facts(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Akari",
                "profile": {
                    "profile_id": "akari_adult",
                    "person_id": "akari",
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["guarded"],
                },
            }
        ],
    )

    manager.merge_global_characters(
        chapter="global_registry",
        characters=[
            {
                "name": "Akari",
                "profile": {
                    "profile_id": "akari_adult",
                    "person_id": "akari",
                    "age_stage": "adult",
                    "gender": "female",
                    "race_or_ethnicity": "Japanese",
                    "accent": "Tokyo",
                    "occupation": "barista",
                    "personality": ["guarded", "wry"],
                },
                "evidence": [{"chapter": "chapter_004", "note": "Workplace scene"}],
            }
        ],
    )

    registry = read_json(paths.registry)
    identity = registry["characters"]["akari_adult"]["identity_profile"]
    assert identity["race_or_ethnicity"] == "Japanese"
    assert identity["accent"] == "Tokyo"
    assert identity["occupation"] == "barista"
    assert identity["personality"] == ["guarded", "wry"]


def test_resolve_effective_voice_matches_unique_short_display_name():
    registry = {
        "book": {"slug": "demo"},
        "narrator": {"role_id": "narrator", "display_name": "Narrator"},
        "characters": {
            "buddy_waleski_adult": {
                "role_id": "buddy_waleski_adult",
                "display_name": "Buddy Waleski",
                "aliases": [],
                "voice_variants": {
                    "default": {
                        "role_id": "buddy_waleski_adult_default",
                        "display_name": "Buddy Waleski_default",
                        "voice_profile": {"description": "adult male", "qwen_instruct": "adult male"},
                        "voice_config_path": None,
                    }
                },
            }
        },
    }

    effective = resolve_effective_voice(registry, "Buddy", "dialogue")

    assert effective["role_id"] == "buddy_waleski_adult_default"
