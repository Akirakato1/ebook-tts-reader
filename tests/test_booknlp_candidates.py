from ebook_tts_pipeline.annotation.booknlp_artifacts import BookNlpQuoteRow
from ebook_tts_pipeline.annotation.booknlp_candidates import map_booknlp_quotes_to_extraction
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue


def test_map_booknlp_quote_to_matching_extracted_quote_id():
    text = 'Mary paused. "The apple of my eye," Mr. Pounds said.'
    extraction = extract_quoted_dialogue(text)
    rows = [
        BookNlpQuoteRow(
            quote_start_token=3,
            quote_end_token=9,
            mention_start_token=10,
            mention_end_token=12,
            mention_phrase="Mr. Pounds",
            character_id="7",
            quote_text="The apple of my eye,",
        )
    ]

    candidates = map_booknlp_quotes_to_extraction("chapter_017", extraction, rows)

    assert len(candidates) == 1
    assert candidates[0].quote_idx == 1
    assert candidates[0].quote_id == "q001"
    assert candidates[0].booknlp_character_id == "7"
    assert candidates[0].mention_phrase == "Mr. Pounds"
