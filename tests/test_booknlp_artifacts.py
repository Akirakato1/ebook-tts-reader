from ebook_tts_pipeline.annotation.booknlp_artifacts import parse_booknlp_quotes, stitch_chapters_for_booknlp


def test_parse_booknlp_quotes_reads_core_speaker_fields(tmp_path):
    quotes_path = tmp_path / "demo.quotes"
    quotes_path.write_text(
        "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n"
        "10\t12\t13\t14\tMr. Pounds\t7\tThe apple of my eye.\n",
        encoding="utf-8",
    )

    rows = parse_booknlp_quotes(quotes_path)

    assert len(rows) == 1
    assert rows[0].quote_start_token == 10
    assert rows[0].quote_end_token == 12
    assert rows[0].mention_phrase == "Mr. Pounds"
    assert rows[0].character_id == "7"
    assert rows[0].quote_text == "The apple of my eye."


def test_stitch_chapters_records_char_offsets():
    stitched = stitch_chapters_for_booknlp(
        {
            "chapter_001": "One.",
            "chapter_002": "Two.",
        }
    )

    assert stitched.text == "[chapter_001]\nOne.\n\n[chapter_002]\nTwo."
    assert stitched.chapter_offsets["chapter_001"].content_start == len("[chapter_001]\n")
    assert stitched.chapter_offsets["chapter_001"].content_end == len("[chapter_001]\nOne.")
    assert stitched.chapter_offsets["chapter_002"].content_start == stitched.text.index("Two.")
