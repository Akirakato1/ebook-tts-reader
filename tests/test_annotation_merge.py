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
