from ebook_tts_pipeline.paths import BookPaths
from ebook_tts_pipeline.read_along.audiobook import (
    AUDIOBOOK_WINDOW_PROFILES,
    build_audiobook_windows,
)


def test_book_paths_exposes_audiobook_artifacts(tmp_path):
    paths = BookPaths(tmp_path / "book")

    assert paths.audiobook_chapter_audio("chapter_015") == (
        tmp_path / "book" / "audiobook" / "chapter_015.wav"
    )
    assert paths.audiobook_chapter_timeline("chapter_015") == (
        tmp_path / "book" / "audiobook" / "chapter_015.timeline.json"
    )
    assert paths.audiobook_manifest == tmp_path / "book" / "audiobook" / "manifest.json"
    assert paths.audiobook_settings == tmp_path / "book" / "audiobook" / "settings.json"
    assert paths.audiobook_position == tmp_path / "book" / "audiobook" / "position.json"
    assert paths.audiobook_narrator_profile == tmp_path / "book" / "audiobook" / "narrator_profile.json"


def test_build_audiobook_windows_uses_larger_character_windows_and_preserves_unit_order():
    units = [
        _unit(0, "Narrator", "narrator", "A" * 35),
        _unit(1, "Leigh", "leigh_adult", "B" * 35),
        _unit(2, "Narrator", "narrator", "C" * 35),
        _unit(3, "Callie", "callie_adult", "D" * 35),
    ]

    windows = build_audiobook_windows(units, max_chars=120, max_roles=2)

    assert [[job["unit_idx"] for job in window] for window in windows] == [[0, 1], [2, 3]]
    assert [[job["role_id"] for job in window] for window in windows] == [
        ["narrator", "leigh_adult"],
        ["narrator", "callie_adult"],
    ]


def test_audiobook_balanced_window_profile_is_larger_than_live_readalong_units():
    profile = AUDIOBOOK_WINDOW_PROFILES["balanced"]

    assert profile["max_chars"] >= 4000
    assert profile["max_roles"] >= 4


def _unit(unit_id, role, role_id, text):
    return {
        "chapter": "chapter_001",
        "unit_id": unit_id,
        "text": text,
        "source_start": unit_id * 10,
        "source_end": unit_id * 10 + len(text),
        "role": role,
        "role_id": role_id,
        "type": "narration" if role_id == "narrator" else "dialogue",
        "voice_config_path": f"voices/{role_id}.qvp",
        "quote_id": None,
        "sentence_idx": unit_id,
        "character": None if role_id == "narrator" else role,
        "voice_variant": None,
    }
