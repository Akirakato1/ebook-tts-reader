from ebook_tts_pipeline.annotation.quote_attribution import QuoteAttributionResult
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.domain import AnnotationResult, Sentence, SentenceArtifact, SentenceUnit
from ebook_tts_pipeline.tts.script import build_tts_script, build_tts_script_from_quotes, render_qwen_dialogue_script


def test_builds_quote_attributed_tts_script_with_single_voice_roles():
    chapter_text = 'Callie said, "Stay here." She left.'
    extraction = extract_quoted_dialogue(chapter_text)
    attribution = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child"],
            "quotes": [[1, 0, "dialogue"]],
        }
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
                "voice_profile": {"qwen_instruct": "Callie child voice."},
                "voice_config_path": "voices/callie_child.qvp",
            }
        },
    }

    script = build_tts_script_from_quotes(
        chapter="chapter_001",
        chapter_text=chapter_text,
        extraction=extraction,
        attribution=attribution,
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
            "text": "Callie said,",
            "voice_config_path": "voices/narrator.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "callie_child",
            "role_id": "callie_child",
            "character": "Callie",
            "type": "dialogue",
            "text": '"Stay here."',
            "voice_config_path": "voices/callie_child.qvp",
        },
        {
            "sentence_idx": 2,
            "unit_idx": 2,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "She left.",
            "voice_config_path": "voices/narrator.qvp",
        },
    ]
    assert script.qwen_dialogue_text == (
        "Narrator: Callie said,\n"
        'callie_child: "Stay here."\n'
        "Narrator: She left."
    )


def test_quote_tts_script_splits_narrator_spans_into_sentence_jobs():
    chapter_text = '"Ready?" Callie stood up. She checked the exit. "Ready."'
    extraction = extract_quoted_dialogue(chapter_text)
    attribution = QuoteAttributionResult.from_dict(
        {
            "roles": ["callie_child"],
            "quotes": [[1, 0], [2, 0]],
        }
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
                "voice_profile": {"qwen_instruct": "Callie child voice."},
                "voice_config_path": "voices/callie_child.qvp",
            }
        },
    }

    script = build_tts_script_from_quotes(
        chapter="chapter_001",
        chapter_text=chapter_text,
        extraction=extraction,
        attribution=attribution,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [(job.role, job.text) for job in script.jobs] == [
        ("callie_child", '"Ready?"'),
        ("Narrator", "Callie stood up."),
        ("Narrator", "She checked the exit."),
        ("callie_child", '"Ready."'),
    ]
    assert [job.sentence_idx for job in script.jobs] == [0, 1, 2, 3]


def test_tts_script_builds_role_tagged_qwen_sections_from_annotation_and_sentences():
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
                    "voice_config_path": "voices/elena.qvp",
                    "voice_profile": {"qwen_instruct": "Elena voice."},
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
            "role": "elena",
            "role_id": "elena",
            "character": "Elena",
            "type": "dialogue",
            "text": '"Hello,"',
            "voice_config_path": "voices/elena.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Elena said.",
            "voice_config_path": "voices/narrator.qvp",
        },
        {
            "sentence_idx": 2,
            "unit_idx": 2,
            "role": "elena",
            "role_id": "elena",
            "character": "Elena",
            "type": "thought",
            "text": "She left.",
            "voice_config_path": "voices/elena.qvp",
        },
    ]
    window = script.windows[0].to_dict()
    assert "batches" not in window
    assert window["qwen_text"] == (
        "Narrator: It rained.\n"
        'elena: "Hello,"\n'
        "Narrator: Elena said.\n"
        "elena: She left."
    )
    assert script.qwen_dialogue_text == (
        "Narrator: It rained.\n"
        'elena: "Hello,"\n'
        "Narrator: Elena said.\n"
        "elena: She left."
    )


def test_tts_script_readds_role_tags_when_sections_split_inside_same_role():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[
            Sentence(0, "First long line."),
            Sentence(1, "Second long line."),
        ],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (0, 0, 1)],
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
        max_chars=28,
        max_roles=8,
        language="auto",
    )

    assert [window.qwen_text for window in script.windows] == [
        "Narrator: First long line.",
        "Narrator: Second long line.",
    ]


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
                    "voice_config_path": "voices/callie_child.qvp",
                    "voice_profile": {"qwen_instruct": "Callie child voice."},
                },
                "callie_adult": {
                    "role_id": "callie_adult",
                    "display_name": "Callie",
                    "aliases": ["Callie adult"],
                    "voice_config_path": "voices/callie_adult.qvp",
                    "voice_profile": {"qwen_instruct": "Callie adult voice."},
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
            "role": "callie_child",
            "role_id": "callie_child",
            "character": "Callie",
            "type": "dialogue",
            "text": '"Stay here,"',
            "voice_config_path": "voices/callie_child.qvp",
        },
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Callie said.",
            "voice_config_path": "voices/narrator.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "callie_adult",
            "role_id": "callie_adult",
            "character": "Callie",
            "type": "thought",
            "text": '"I remember,"',
            "voice_config_path": "voices/callie_adult.qvp",
        },
        {
            "sentence_idx": 1,
            "unit_idx": 1,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Callie thought.",
            "voice_config_path": "voices/narrator.qvp",
        },
    ]
    assert script.qwen_dialogue_text == (
        'callie_child: "Stay here,"\n'
        "Narrator: Callie said.\n"
        'callie_adult: "I remember,"\n'
        "Narrator: Callie thought."
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
                    "voice_config_path": "voices/callie_child.qvp",
                    "voice_profile": {"qwen_instruct": "Callie child voice."},
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
            "role": "callie_child",
            "role_id": "callie_child",
            "character": "Callie",
            "type": "dialogue",
            "text": '"Stay here,"',
            "voice_config_path": "voices/callie_child.qvp",
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
    assert script.qwen_dialogue_text == 'callie_child: "Stay here,"\nNarrator: Callie said.'


def test_tts_script_extracts_narrator_context_after_annotation_without_changing_quote_speaker():
    artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, 'Walter said, "I like your jacket."')],
        units=[SentenceUnit(0, 0, 'Walter said, "I like your jacket."')],
    )
    annotation = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Walter"],
        types=["narration", "dialogue", "thought"],
        script=[(1, 1, 0)],
    )
    registry = {
        "book": {"slug": "demo"},
        "narrator": {
            "role_id": "narrator",
            "display_name": "Narrator",
            "voice_config_path": "voices/narrator.qvp",
        },
        "characters": {
                "walter": {
                    "role_id": "walter",
                    "display_name": "Walter",
                    "aliases": [],
                    "voice_config_path": "voices/walter.qvp",
                    "voice_profile": {"qwen_instruct": "Walter voice."},
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

    assert [(job.role, job.type, job.text) for job in script.jobs] == [
        ("Narrator", "narration", "Walter said,"),
        ("walter", "dialogue", '"I like your jacket."'),
    ]

    continuation_artifact = SentenceArtifact(
        chapter="chapter_001",
        source_path="chapters/chapter_001.txt",
        segmenter={"name": "test"},
        sentences=[Sentence(0, "You did what she told you to do.\u201d It was almost a relief.")],
        units=[SentenceUnit(0, 0, "You did what she told you to do.\u201d It was almost a relief.")],
    )

    continuation_script = build_tts_script(
        chapter="chapter_001",
        annotation=annotation,
        artifact=continuation_artifact,
        registry=registry,
        max_chars=1000,
        max_roles=8,
        language="auto",
    )

    assert [(job.role, job.type, job.text) for job in continuation_script.jobs] == [
        ("walter", "dialogue", "You did what she told you to do.\u201d"),
        ("Narrator", "narration", "It was almost a relief."),
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
                "role_id": "chapter_013_tmp_001",
                "voice_config_path": "voices/_temp/chapter_013/tmp_001.qvp",
                "voice_profile": {"qwen_instruct": "An adult male security guard voice."},
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
            "role": "chapter_013_tmp_001",
            "role_id": "chapter_013_tmp_001",
            "character": "Security Guard",
            "type": "dialogue",
            "text": '"Move along,"',
            "voice_config_path": "voices/_temp/chapter_013/tmp_001.qvp",
        },
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "the guard said.",
            "voice_config_path": None,
        }
    ]
    assert script.qwen_dialogue_text == (
        'chapter_013_tmp_001: "Move along,"\n'
        "Narrator: the guard said."
    )


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


def test_render_qwen_dialogue_script_speaks_hash_symbol_as_hashtag():
    script = render_qwen_dialogue_script(
        [
            {
                "sentence_idx": 0,
                "role": "Narrator",
                "type": "narration",
                "text": "Follow #TeamCallie.",
            }
        ]
    )

    assert script == "Narrator: Follow hashtag TeamCallie."
