from pathlib import Path

from ebook_tts_pipeline.tts.wsl_paths import to_wsl_path, translate_job_paths


def test_to_wsl_path_translates_windows_drive_path():
    assert (
        to_wsl_path(Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\models\qwen-tts"))
        == "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/models/qwen-tts"
    )


def test_to_wsl_path_leaves_relative_path_posix():
    assert to_wsl_path(Path("voices/narrator.qvp")) == "voices/narrator.qvp"


def test_translate_job_paths_resolves_voice_path_against_book_root():
    jobs = [
        {
            "sentence_idx": 0,
            "unit_idx": 0,
            "role": "Narrator",
            "role_id": "narrator",
            "type": "narration",
            "text": "Hello.",
            "voice_config_path": "voices/narrator.qvp",
        }
    ]

    translated = translate_job_paths(
        jobs,
        book_root=Path(r"C:\Users\zhuyl\OneDrive\Documents\Ebook Reader\test book"),
    )

    assert (
        translated[0]["voice_config_path"]
        == "/mnt/c/Users/zhuyl/OneDrive/Documents/Ebook Reader/test book/voices/narrator.qvp"
    )
