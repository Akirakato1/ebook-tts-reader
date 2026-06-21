import pytest

from ebook_tts_pipeline.annotation.validator import AnnotationValidationError, validate_annotation
from ebook_tts_pipeline.domain import AnnotationResult, Sentence
from ebook_tts_pipeline.windowing import build_llm_windows, build_tts_windows


def test_llm_windowing_moves_sentence_to_next_window_when_it_would_exceed_limit():
    sentences = [
        Sentence(idx=0, text="aaaa"),
        Sentence(idx=1, text="bbbb"),
        Sentence(idx=2, text="cccc"),
    ]

    windows = build_llm_windows(sentences, max_chars=9)

    assert [[sentence.idx for sentence in window.sentences] for window in windows] == [[0, 1], [2]]


def test_tts_windowing_respects_eight_role_limit_and_sentence_atomicity():
    jobs = [
        {"sentence_idx": idx, "role": f"Role{idx}", "text": "Hi."}
        for idx in range(9)
    ]

    windows = build_tts_windows(jobs, max_chars=1000, max_roles=8)

    assert [len({job["role"] for job in window.jobs}) for window in windows] == [8, 1]
    assert [window.jobs[0]["sentence_idx"] for window in windows] == [0, 8]


def test_annotation_validator_accepts_complete_compact_script():
    result = AnnotationResult(
        new_characters=[],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )

    validate_annotation(result, expected_sentence_indices=[0, 1], known_names={"Elena"})


def test_annotation_validator_rejects_duplicate_sentence_ids():
    result = AnnotationResult(
        new_characters=[],
        roles=["Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (0, 0, 0)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0, 1], known_names=set())

    assert "missing sentence indexes: [1]" in str(exc.value)
    assert "duplicate sentence indexes: [0]" in str(exc.value)


def test_annotation_validator_rejects_new_character_alias_collision():
    result = AnnotationResult(
        new_characters=[{"name": "Elena", "profile": {}, "voice": {}}],
        roles=["Narrator", "Elena"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0, 1], known_names={"elena"})

    assert "collides with existing character or alias: Elena" in str(exc.value)


def test_annotation_validator_rejects_narrator_as_new_character():
    result = AnnotationResult(
        new_characters=[{"name": "Narrator", "profile": {}, "voice": {}}],
        roles=["Narrator"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0], known_names={"Narrator"})

    assert "collides with existing character or alias: Narrator" in str(exc.value)


def test_annotation_validator_rejects_malformed_new_character_voice_profile():
    result = AnnotationResult(
        new_characters=[{"name": "Leigh", "profile": {}, "voice": "warm adult woman"}],
        roles=["Narrator", "Leigh"],
        types=["narration", "dialogue", "thought"],
        script=[(0, 0, 0), (1, 1, 1)],
    )

    with pytest.raises(AnnotationValidationError) as exc:
        validate_annotation(result, expected_sentence_indices=[0, 1], known_names={"Narrator"})

    assert "new character voice must be an object: Leigh" in str(exc.value)
