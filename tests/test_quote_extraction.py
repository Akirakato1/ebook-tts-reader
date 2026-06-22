from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue


def test_quote_extractor_splits_adjacent_smart_quotes_from_narration():
    text = (
        "Callie smiled. "
        "\u201cI found this for you in the return bin.\u201d "
        "\u201cWonderful, thank you.\u201d Callie took the thick paperback."
    )

    extraction = extract_quoted_dialogue(text)

    assert [quote.to_dict() for quote in extraction.quotes] == [
        {
            "idx": 1,
            "quote_id": "q001",
            "start": text.index("\u201cI found"),
            "end": text.index("\u201d ", text.index("\u201cI found")) + 1,
            "text": "\u201cI found this for you in the return bin.\u201d",
        },
        {
            "idx": 2,
            "quote_id": "q002",
            "start": text.index("\u201cWonderful"),
            "end": text.index("\u201d Callie") + 1,
            "text": "\u201cWonderful, thank you.\u201d",
        },
    ]
    assert [span.to_dict() for span in extraction.narrator_spans] == [
        {
            "idx": 1,
            "start": 0,
            "end": len("Callie smiled. "),
            "text": "Callie smiled.",
        },
        {
            "idx": 2,
            "start": text.index(" Callie took"),
            "end": len(text),
            "text": "Callie took the thick paperback.",
        },
    ]


def test_quote_extractor_preserves_straight_quote_offsets_and_tags():
    text = 'Walter said, "I like your jacket." Then he added, "Keep it."'

    extraction = extract_quoted_dialogue(text)

    assert [(quote.quote_id, quote.text, text[quote.start:quote.end]) for quote in extraction.quotes] == [
        ("q001", '"I like your jacket."', '"I like your jacket."'),
        ("q002", '"Keep it."', '"Keep it."'),
    ]
    assert [span.text for span in extraction.narrator_spans] == [
        "Walter said,",
        "Then he added,",
    ]


def test_quote_extractor_marks_quotes_in_original_chapter_text():
    text = 'One. "Two." Three.'

    extraction = extract_quoted_dialogue(text)

    assert extraction.to_marked_text() == 'One. |q001| "Two." ||q001|| Three.'
