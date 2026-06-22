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


def test_tts_script_normalizes_haiku_mislabeled_mixed_quote_units():
    artifact = SentenceArtifact(
        chapter="chapter_013",
        source_path="chapters/chapter_013.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(193, "She left three seats between her and Walter when she sat down."),
            Sentence(194, "He said, \u201cWelcome, friend.\u201d Callie peeled off her mask."),
        ],
        units=[
            SentenceUnit(202, 193, "She left three seats between her and Walter when she sat down."),
            SentenceUnit(203, 194, "He said,"),
            SentenceUnit(204, 194, "\u201cWelcome, friend.\u201d"),
            SentenceUnit(205, 194, "Callie peeled off her mask."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Callie adult", "Walter Collier adult"],
        types=["narration", "dialogue", "thought"],
        script=[
            (0, 0, 202),
            (2, 1, 203),
            (1, 0, 204),
            (1, 0, 205),
        ],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "callie_adult": {
                "role_id": "callie_adult",
                "display_name": "Callie",
                "aliases": ["Callie adult"],
                "identity_profile": {"gender": "female"},
                "voice_variants": {
                    "default": {
                        "role_id": "callie_adult_default",
                        "display_name": "Callie_default",
                        "voice_config_path": "voices/callie_adult_default.qvp",
                        "voice_profile": {"qwen_instruct": "Callie adult aloud."},
                    }
                },
            },
            "walter_collier_adult": {
                "role_id": "walter_collier_adult",
                "display_name": "Walter Collier",
                "aliases": ["Walter Collier adult"],
                "identity_profile": {"gender": "male"},
                "voice_variants": {
                    "default": {
                        "role_id": "walter_collier_adult_default",
                        "display_name": "Walter Collier_default",
                        "voice_config_path": "voices/walter_collier_adult_default.qvp",
                        "voice_profile": {"qwen_instruct": "Walter adult aloud."},
                    }
                },
            },
        },
    }

    script = build_tts_script(
        chapter="chapter_013",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [
        (job.unit_idx, job.role, job.type, job.text)
        for job in script.jobs
        if job.unit_idx in {203, 204, 205}
    ] == [
        (203, "Narrator", "narration", "He said,"),
        (204, "Walter Collier_default", "dialogue", "\u201cWelcome, friend.\u201d"),
        (205, "Narrator", "narration", "Callie peeled off her mask."),
    ]


def test_tts_script_splits_quote_continuations_inside_annotated_units():
    artifact = SentenceArtifact(
        chapter="chapter_013",
        source_path="chapters/chapter_013.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(343, "\u201cI couldn\u2019t see your face in the video."),
            Sentence(344, "You never looked up."),
            Sentence(345, "You just did what Harleigh told you to do.\u201d It was almost a relief."),
            Sentence(401, "He told the guard, \u201cNone of this will look good for the school, will it?"),
            Sentence(402, "And it won\u2019t look good for you.\u201d This seemed to sway the guard."),
        ],
        units=[
            SentenceUnit(391, 343, "\u201cI couldn\u2019t see your face in the video."),
            SentenceUnit(392, 344, "You never looked up."),
            SentenceUnit(393, 345, "You just did what Harleigh told you to do.\u201d It was almost a relief."),
            SentenceUnit(471, 401, "He told the guard, \u201cNone of this will look good for the school, will it?"),
            SentenceUnit(472, 402, "And it won\u2019t look good for you.\u201d This seemed to sway the guard."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Andrew Tenant adult"],
        types=["narration", "dialogue", "thought"],
        script=[
            (1, 1, 391),
            (0, 0, 392),
            (1, 1, 393),
            (1, 1, 471),
            (0, 0, 472),
        ],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "andrew_tenant_adult": {
                "role_id": "andrew_tenant_adult",
                "display_name": "Andrew Tenant",
                "aliases": ["Andrew Tenant adult"],
                "identity_profile": {"gender": "male"},
                "voice_variants": {
                    "default": {
                        "role_id": "andrew_tenant_adult_default",
                        "display_name": "Andrew Tenant_default",
                        "voice_config_path": "voices/andrew_tenant_adult_default.qvp",
                        "voice_profile": {"qwen_instruct": "Andrew adult aloud."},
                    }
                },
            },
        },
    }

    script = build_tts_script(
        chapter="chapter_013",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [(job.unit_idx, job.role, job.type, job.text) for job in script.jobs] == [
        (
            391,
            "Andrew Tenant_default",
            "dialogue",
            "\u201cI couldn\u2019t see your face in the video.",
        ),
        (392, "Andrew Tenant_default", "dialogue", "You never looked up."),
        (393, "Andrew Tenant_default", "dialogue", "You just did what Harleigh told you to do.\u201d"),
        (393, "Narrator", "narration", "It was almost a relief."),
        (471, "Narrator", "narration", "He told the guard,"),
        (
            471,
            "Andrew Tenant_default",
            "dialogue",
            "\u201cNone of this will look good for the school, will it?",
        ),
        (472, "Andrew Tenant_default", "dialogue", "And it won\u2019t look good for you.\u201d"),
        (472, "Narrator", "narration", "This seemed to sway the guard."),
    ]


def test_tts_script_uses_local_speaker_for_pronoun_tag_with_intervening_adverb():
    artifact = SentenceArtifact(
        chapter="chapter_013",
        source_path="chapters/chapter_013.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(401, "He told the guard, \u201cNone of this will look good?"),
            Sentence(
                402,
                "And it won\u2019t look good for you.\u201d This seemed to sway the guard, "
                "but he still asked, \u201cAre you sure?\u201d",
            ),
        ],
        units=[
            SentenceUnit(471, 401, "He told the guard, \u201cNone of this will look good?"),
            SentenceUnit(
                472,
                402,
                "And it won\u2019t look good for you.\u201d This seemed to sway the guard, but he still asked,",
            ),
            SentenceUnit(473, 402, "\u201cAre you sure?\u201d"),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_005",
                "label": "Security Guard",
                "profile": {"age_stage": "adult", "gender": "male", "personality": ["watchful"]},
            }
        ],
        roles=["Narrator", "Andrew Tenant adult", "tmp_005"],
        types=["narration", "dialogue", "thought"],
        script=[
            (1, 1, 471),
            (0, 0, 472),
            (0, 1, 473),
        ],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
            "andrew_tenant_adult": {
                "role_id": "andrew_tenant_adult",
                "display_name": "Andrew Tenant",
                "aliases": ["Andrew Tenant adult"],
                "identity_profile": {"gender": "male"},
                "voice_variants": {
                    "default": {
                        "role_id": "andrew_tenant_adult_default",
                        "display_name": "Andrew Tenant_default",
                        "voice_config_path": "voices/andrew_tenant_adult_default.qvp",
                        "voice_profile": {"qwen_instruct": "Andrew adult aloud."},
                    }
                },
            },
        },
    }
    temp_registry = {
        "chapter": "chapter_013",
        "speakers": {
            "tmp_005": {
                "local_id": "tmp_005",
                "label": "Security Guard",
                "voice_variants": {
                    "default": {
                        "role_id": "chapter_013_tmp_005_default",
                        "display_name": "Security Guard_default",
                        "voice_config_path": "voices/_temp/chapter_013/tmp_005_default.qvp",
                        "voice_profile": {"qwen_instruct": "Security guard aloud."},
                    }
                },
            }
        },
    }

    script = build_tts_script(
        chapter="chapter_013",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
        temp_registry=temp_registry,
    )

    assert [(job.unit_idx, job.role, job.type, job.text) for job in script.jobs if job.unit_idx == 473] == [
        (473, "Security Guard_default", "dialogue", "\u201cAre you sure?\u201d")
    ]


def test_tts_script_resolves_chapter_local_speakers_from_temp_registry():
    artifact = SentenceArtifact(
        chapter="chapter_013",
        source_path="chapters/chapter_013.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, '"Move along," the guard said.')],
    )
    annotation = AnnotationResult(
        new_characters=[],
        local_speakers=[
            {
                "local_id": "tmp_001",
                "label": "Security Guard",
                "profile": {
                    "age_stage": "adult",
                    "gender": "male",
                    "personality": ["authoritative"],
                    "occupation": "security guard",
                },
            }
        ],
        roles=["tmp_001"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 1, 0)],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {"role_id": "narrator", "display_name": "Narrator"},
        "characters": {},
    }
    temp_registry = {
        "chapter": "chapter_013",
        "speakers": {
            "tmp_001": {
                "local_id": "tmp_001",
                "label": "Security Guard",
                "voice_variants": {
                    "default": {
                        "role_id": "chapter_013_tmp_001_default",
                        "display_name": "Security Guard_default",
                        "voice_config_path": "voices/_temp/chapter_013/tmp_001_default.qvp",
                        "voice_profile": {"qwen_instruct": "An adult male security guard voice."},
                    }
                },
            }
        },
    }

    script = build_tts_script(
        chapter="chapter_013",
        annotation=annotation,
        artifact=artifact,
        registry=registry,
        temp_registry=temp_registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [job.to_dict() for job in script.jobs] == [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Security Guard_default",
            "role_id": "chapter_013_tmp_001_default",
            "character": "Security Guard",
            "voice_variant": "default",
            "type": "dialogue",
            "text": '"Move along," the guard said.',
            "voice_config_path": "voices/_temp/chapter_013/tmp_001_default.qvp",
        }
    ]
    assert script.qwen_dialogue_text == 'Security Guard_default: "Move along," the guard said.'


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
