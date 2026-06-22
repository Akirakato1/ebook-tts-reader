from ebook_tts_pipeline.annotation.merge import merge_annotation_windows
from ebook_tts_pipeline.domain import AnnotationResult


def test_merge_annotation_windows_preserves_new_characters_and_reindexes_roles():
    registry = {
        "characters": {
            "elena": {
                "role_id": "elena",
                "display_name": "Elena",
                "aliases": [],
            }
        }
    }
    first = AnnotationResult(
        new_characters=[
            {
                "name": "Elena",
                "profile": {"age_range": "young adult"},
                "voice": {
                    "description": "soft young woman",
                    "qwen_instruct": "A soft young adult female voice.",
                },
            }
        ],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )
    second = AnnotationResult(
        new_characters=[],
        roles=["elena", "Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 2, 2), (1, 0, 3)],
    )

    merged = merge_annotation_windows([first, second], registry)

    assert [character["name"] for character in merged.new_characters] == ["Elena"]
    assert merged.roles == ["Narrator", "Elena"]
    assert merged.script == [(0, 0, 0), (1, 1, 1), (1, 2, 2), (0, 0, 3)]


def test_merge_annotation_windows_preserves_disambiguating_alias_for_duplicate_display_names():
    registry = {
        "characters": {
            "callie_child": {
                "role_id": "callie_child",
                "display_name": "Callie",
                "age_stage": "child",
                "aliases": ["Callie child"],
            },
            "callie_adult": {
                "role_id": "callie_adult",
                "display_name": "Callie",
                "age_stage": "adult",
                "aliases": ["Callie adult"],
            },
        }
    }
    first = AnnotationResult(
        new_characters=[],
        roles=["Callie child"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0)],
    )
    second = AnnotationResult(
        new_characters=[],
        roles=["callie_adult"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 2, 1)],
    )

    merged = merge_annotation_windows([first, second], registry)

    assert merged.roles == ["Callie child", "Callie adult"]
    assert merged.script == [(0, 1, 0), (1, 2, 1)]


def test_merge_annotation_windows_preserves_local_speakers_without_global_promotion():
    registry = {"characters": {}}
    first = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_001",
                "label": "Security Guard",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["authoritative"],
                },
            }
        ],
        roles=["Narrator", "tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )
    second = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_002",
                "label": "Houseless man",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["anxious"],
                },
            }
        ],
        roles=["Houseless man"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 2)],
    )

    merged = merge_annotation_windows([first, second], registry)

    assert merged.new_characters == []
    assert [speaker["label"] for speaker in merged.local_speakers] == ["Security Guard", "Houseless man"]
    assert merged.roles == ["Narrator", "tmp_001", "tmp_002"]
    assert merged.script == [(0, 0, 0), (1, 1, 1), (2, 1, 2)]


def test_merge_annotation_windows_renumbers_reused_local_ids_from_different_windows():
    registry = {"characters": {}}
    first = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_001",
                "label": "Security Guard",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["authoritative"],
                },
            }
        ],
        roles=["tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0)],
    )
    second = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_001",
                "label": "Houseless man",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["anxious"],
                },
            }
        ],
        roles=["tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 1)],
    )

    merged = merge_annotation_windows([first, second], registry)

    assert [(speaker["local_id"], speaker["label"]) for speaker in merged.local_speakers] == [
        ("tmp_001", "Security Guard"),
        ("tmp_002", "Houseless man"),
    ]
    assert merged.roles == ["tmp_001", "tmp_002"]
    assert merged.script == [(0, 1, 0), (1, 1, 1)]
