from ebook_tts_pipeline.domain import AnnotationResult, Sentence, SentenceArtifact, SentenceUnit
from ebook_tts_pipeline.tts.script import build_tts_script, render_qwen_dialogue_script


def test_tts_script_builds_qwen_batches_from_annotation_and_sentences():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(0, "It rained."),
            Sentence(1, '"Hello," Elena said.'),
            Sentence(2, "She left."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1), (1, 2, 2)],
    )
    registry = {
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "elena": {
                "role_id": "elena",
                "display_name": "Elena",
                "aliases": [],
                "voice_variants": {
                    "default": {
                        "role_id": "elena_default",
                        "display_name": "Elena_default",
                        "voice_config_path": "voices/elena_default.qvp",
                        "voice_profile": {"qwen_instruct": "Elena aloud."},
                    },
                    "internal": {
                        "role_id": "elena_internal",
                        "display_name": "Elena_internal",
                        "voice_config_path": "voices/elena_internal.qvp",
                        "voice_profile": {"qwen_instruct": "Elena inward."},
                    },
                },
            }
        },
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [job.to_dict() for job in script.jobs] == [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "It rained.",
            "voice_config_path": "voices/narrator.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "Elena_default",
            "role_id": "elena_default",
            "character": "Elena",
            "voice_variant": "default",
            "type": "dialogue",
            "text": '"Hello," Elena said.',
            "voice_config_path": "voices/elena_default.qvp",
        },
        {
            "sentence_idx": 2,
            "unit_idx": 2,
            "role": "Elena_internal",
            "role_id": "elena_internal",
            "character": "Elena",
            "voice_variant": "internal",
            "type": "thought",
            "text": "She left.",
            "voice_config_path": "voices/elena_internal.qvp",
        },
    ]
    assert [batch.to_dict() for batch in script.windows[0].batches] == [
        {
            "batch_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "voice_config_path": "voices/narrator.qvp",
            "language": "auto",
            "sentence_indices": [0],
            "unit_indices": [0],
            "types": ["narration"],
            "text": ["It rained."],
        },
        {
            "batch_idx": 1,
            "role": "Elena_default",
            "role_id": "elena_default",
            "voice_config_path": "voices/elena_default.qvp",
            "language": "auto",
            "sentence_indices": [1],
            "unit_indices": [1],
            "types": ["dialogue"],
            "text": ['"Hello," Elena said.'],
        },
        {
            "batch_idx": 2,
            "role": "Elena_internal",
            "role_id": "elena_internal",
            "voice_config_path": "voices/elena_internal.qvp",
            "language": "auto",
            "sentence_indices": [2],
            "unit_indices": [2],
            "types": ["thought"],
            "text": ["She left."],
        },
    ]
    assert script.qwen_dialogue_text == (
        "Narrator: It rained.\n"
        'Elena_default: "Hello," Elena said.\n'
        "Elena_internal: She left."
    )


def test_tts_script_resolves_age_stage_aliases_to_unique_voice_roles():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(0, '"Stay here," Callie said.'),
            Sentence(1, '"I remember," Callie thought.'),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Callie child", "Callie adult"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0), (1, 2, 1)],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "callie_child": {
                "role_id": "callie_child",
                "display_name": "Callie",
                "aliases": ["Callie child"],
                "voice_variants": {
                    "default": {
                        "role_id": "callie_child_default",
                        "display_name": "Callie_default",
                        "voice_config_path": "voices/callie_child_default.qvp",
                        "voice_profile": {"qwen_instruct": "Callie child aloud."},
                    },
                    "internal": {
                        "role_id": "callie_child_internal",
                        "display_name": "Callie_internal",
                        "voice_config_path": "voices/callie_child_internal.qvp",
                        "voice_profile": {"qwen_instruct": "Callie child inward."},
                    },
                },
            },
            "callie_adult": {
                "role_id": "callie_adult",
                "display_name": "Callie",
                "aliases": ["Callie adult"],
                "voice_variants": {
                    "default": {
                        "role_id": "callie_adult_default",
                        "display_name": "Callie_default",
                        "voice_config_path": "voices/callie_adult_default.qvp",
                        "voice_profile": {"qwen_instruct": "Callie adult aloud."},
                    },
                    "internal": {
                        "role_id": "callie_adult_internal",
                        "display_name": "Callie_internal",
                        "voice_config_path": "voices/callie_adult_internal.qvp",
                        "voice_profile": {"qwen_instruct": "Callie adult inward."},
                    },
                },
            },
        },
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [job.to_dict() for job in script.jobs] == [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "callie_child_default",
            "role_id": "callie_child_default",
            "character": "Callie",
            "voice_variant": "default",
            "type": "dialogue",
            "text": '"Stay here," Callie said.',
            "voice_config_path": "voices/callie_child_default.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "callie_adult_internal",
            "role_id": "callie_adult_internal",
            "character": "Callie",
            "voice_variant": "internal",
            "type": "thought",
            "text": '"I remember," Callie thought.',
            "voice_config_path": "voices/callie_adult_internal.qvp",
        },
    ]
    assert script.qwen_dialogue_text == (
        'callie_child_default: "Stay here," Callie said.\n'
        'callie_adult_internal: "I remember," Callie thought.'
    )


def test_tts_script_uses_annotation_units_for_embedded_dialogue_tags():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, '"Stay here," Callie said.')],
        units=[
            SentenceUnit(0, 0, '"Stay here,"'),
            SentenceUnit(1, 0, "Callie said."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Callie child"],
        types=["narration", "dialogue", "thought"],
        script=[(1, 1, 0), (0, 0, 1)],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "callie_child": {
                "role_id": "callie_child",
                "display_name": "Callie",
                "aliases": ["Callie child"],
                "voice_variants": {
                    "default": {
                        "role_id": "callie_child_default",
                        "display_name": "Callie_default",
                        "voice_config_path": "voices/callie_child_default.qvp",
                        "voice_profile": {"qwen_instruct": "Callie child aloud."},
                    }
                },
            },
        },
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [job.to_dict() for job in script.jobs] == [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Callie_default",
            "role_id": "callie_child_default",
            "character": "Callie",
            "voice_variant": "default",
            "type": "dialogue",
            "text": '"Stay here,"',
            "voice_config_path": "voices/callie_child_default.qvp",
        },
        {
            "sentence_idx": 0,
            "unit_idx": 1,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Callie said.",
            "voice_config_path": "voices/narrator.qvp",
        },
    ]
    assert script.qwen_dialogue_text == 'Callie_default: "Stay here,"\nNarrator: Callie said.'


def test_tts_script_respects_role_limit_when_creating_windows():
    sentences = [Sentence(idx, f"Sentence {idx}.") for idx in range(9)]
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=sentences,
    )
    roles = [f"Role {idx}" for idx in range(9)]
    annotation = AnnotationResult(
        new_characters=[],
        roles=roles,
        types=["narration", "dialogue", "thought"],
        script=[(idx, 1, idx) for idx in range(9)],
    )
    registry = {
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            f"role_{idx}": {
                "role_id": f"role_{idx}",
                "display_name": f"Role {idx}",
                "aliases": [],
                "voice_config_path": f"voices/role_{idx}.qvp",
            }
            for idx in range(9)
        },
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [window.role_count for window in script.windows] == [8, 1]
    assert [window.sentence_indices for window in script.windows] == [
        list(range(8)),
        [8],
    ]


def test_tts_script_orders_jobs_by_sentence_index_even_when_annotation_rows_drift():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(0, "First."),
            Sentence(1, "Second."),
            Sentence(2, "Third."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (0, 0, 2), (0, 0, 1)],
    )
    registry = {
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {},
    }

    script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [job.sentence_idx for job in script.jobs] == [0, 1, 2]
    assert [job.text for job in script.jobs] == ["First.", "Second.", "Third."]


def test_render_qwen_dialogue_script_uses_one_role_prefix_per_contiguous_block():
    script = render_qwen_dialogue_script(
        [
            {
                "sentence_idx": 0,
                "role": "Narrator",
                "type": "narration",
                "text": "First sentence.",
            },
            {
                "sentence_idx": 1,
                "role": "Narrator",
                "type": "narration",
                "text": "Second sentence.",
            },
            {
                "sentence_idx": 2,
                "role": "Elena",
                "type": "dialogue",
                "text": "Hello.",
            },
            {
                "sentence_idx": 3,
                "role": "Narrator",
                "type": "narration",
                "text": "Back to narration.",
            },
        ]
    )

    assert script == (
        "Narrator: First sentence. Second sentence.\n"
        "Elena: Hello.\n"
        "Narrator: Back to narration."
    )
