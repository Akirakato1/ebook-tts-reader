from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.registry import RegistryManager, resolve_effective_voice


def test_registry_initializes_specific_narrator_voice(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")

    registry = read_json(paths.registry)
    narrator_voice = registry["narrator"]["voice_profile"]

    assert "adult male" in narrator_voice["description"]
    assert "baritone" in narrator_voice["qwen_instruct"]
    assert "clear audiobook narration" in narrator_voice["qwen_instruct"]


def test_registry_migrates_only_legacy_default_narrator_voice(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    write_json_atomic(
        paths.registry,
        {
            "book": {"title": "Demo", "slug": "demo"},
            "narrator": {
                "role_id": "narrator",
                "display_name": "Narrator",
                "voice_profile": {
                    "description": "calm literary narrator, clear pacing",
                    "qwen_instruct": "A calm literary narrator voice with clear pacing.",
                },
                "voice_config_path": "voices/narrator.qvp",
                "voice_config_hash": "old-profile-hash",
            },
            "characters": {},
        },
    )

    registry = RegistryManager(paths).load()

    narrator_voice = registry["narrator"]["voice_profile"]
    assert "adult male" in narrator_voice["description"]
    assert "baritone" in narrator_voice["qwen_instruct"]
    assert registry["narrator"]["voice_config_hash"] == "old-profile-hash"

    registry["narrator"]["voice_profile"] = {
        "description": "custom narrator",
        "qwen_instruct": "A custom narrator selected by the user.",
    }
    RegistryManager(paths).save(registry)

    reloaded = RegistryManager(paths).load()

    assert reloaded["narrator"]["voice_profile"]["description"] == "custom narrator"


def test_inserted_character_uses_single_voice_record(tmp_path):
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
    assert "voice_variants" not in elena
    assert elena["voice_config_path"] is None
    assert elena["voice_profile"]["qwen_instruct"].startswith("A adult female voice")

    effective = resolve_effective_voice(registry, "Elena", "dialogue")

    assert effective["role"] == "elena_adult"
    assert effective["role_id"] == "elena_adult"
    assert effective["voice_variant"] is None
    assert effective["voice_record"] is elena


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
    assert "age" not in elena
    assert "age" not in elena["identity_profile"]
    assert "voice_variants" not in elena
    assert elena["voice_config_path"] is None
    assert isinstance(elena["voice_identity"]["seed"], int)
    assert elena["voice_identity"]["differentiators"]
    voice_prompt = elena["voice_profile"]["qwen_instruct"]
    assert "adult female" in voice_prompt
    assert "soft, hesitant" in voice_prompt
    assert "first_seen" not in elena
    assert "timeline" not in elena
    assert "same_person_as" not in elena
    assert "character_profile" not in elena
    assert "narrative_notes" not in elena
    assert "global_evidence" not in elena


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
    elena_voice = registry["characters"]["elena_adult"]["voice_profile"]["qwen_instruct"]
    mira_voice = registry["characters"]["mira_adult"]["voice_profile"]["qwen_instruct"]
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
    assert "age" not in teen
    assert "age" not in teen["identity_profile"]
    assert "narrative_notes" not in teen
    assert "timeline" not in teen
    assert "same_person_as" not in adult
    assert "grooming" not in teen["voice_profile"]["qwen_instruct"].lower()
    assert "callie_teen" == teen["role_id"]
    assert "callie_adult" == adult["role_id"]


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
    assert "global_evidence" not in akari


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


def test_global_registry_merge_creates_age_stage_variant_for_same_name(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["protective"],
                },
            }
        ],
    )

    manager.merge_global_characters(
        chapter="global_registry",
        characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "teen",
                    "gender": "female",
                    "occupation": "student",
                    "personality": ["guarded", "timid"],
                },
            }
        ],
    )

    registry = read_json(paths.registry)
    assert set(registry["characters"]) == {"callie_adult", "callie_teen"}
    assert registry["characters"]["callie_adult"]["identity_profile"]["personality"] == ["protective"]
    teen = registry["characters"]["callie_teen"]
    assert teen["person_id"] == "callie"
    assert teen["age_stage"] == "teen"
    assert teen["identity_profile"]["occupation"] == "student"
    assert teen["role_id"] == "callie_teen"


def test_global_registry_merge_updates_same_name_same_age_stage(tmp_path):
    paths = BookPaths(tmp_path / "demo")
    manager = RegistryManager(paths)
    manager.initialize_if_missing(book_title="Demo Book", book_slug="demo")
    manager.add_new_characters(
        chapter="chapter_001",
        new_characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "adult",
                    "gender": "female",
                    "personality": ["protective"],
                },
            }
        ],
    )

    manager.merge_global_characters(
        chapter="global_registry",
        characters=[
            {
                "name": "Callie",
                "profile": {
                    "age_stage": "adult",
                    "gender": "female",
                    "occupation": "lawyer",
                    "personality": ["direct"],
                },
            }
        ],
    )

    registry = read_json(paths.registry)
    assert set(registry["characters"]) == {"callie_adult"}
    identity = registry["characters"]["callie_adult"]["identity_profile"]
    assert identity["occupation"] == "lawyer"
    assert identity["personality"] == ["protective", "direct"]


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

    assert effective["role_id"] == "buddy_waleski_adult"
    assert effective["role"] == "buddy_waleski_adult"
    assert "voice_variants" not in registry["characters"]["buddy_waleski_adult"]
