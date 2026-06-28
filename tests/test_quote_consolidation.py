from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_consolidation import (
    consolidate_candidates_deterministically,
    render_consolidation_prompt,
)


def test_consolidation_maps_unique_short_honorific_to_registry_role():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    result = consolidate_candidates_deterministically(candidates, registry)

    assert result.resolved_quotes == {1: "mr_john_pounds_adult"}
    assert result.unresolved == []


def test_consolidation_leaves_ambiguous_short_honorific_unresolved():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": [],
            },
            "mr_james_pounds_adult": {
                "role_id": "mr_james_pounds_adult",
                "display_name": "Mr James Pounds",
                "age_stage": "adult",
                "aliases": [],
            },
        }
    }
    candidates = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    result = consolidate_candidates_deterministically(candidates, registry)

    assert result.resolved_quotes == {}
    assert result.unresolved[0].quote_id == "q001"
    assert sorted(result.unresolved[0].candidate_role_ids) == [
        "mr_james_pounds_adult",
        "mr_john_pounds_adult",
    ]


def test_render_consolidation_prompt_uses_compact_quote_table_not_full_chapter():
    registry = {
        "characters": {
            "mr_john_pounds_adult": {
                "role_id": "mr_john_pounds_adult",
                "display_name": "Mr John Pounds",
                "age_stage": "adult",
                "aliases": ["Mr John Pounds adult"],
            }
        }
    }
    unresolved = [
        QuoteAttributionCandidate(
            chapter="chapter_017",
            quote_idx=1,
            quote_id="q001",
            quote_text='"The apple of my eye."',
            booknlp_character_id="7",
            mention_phrase="Mr. Pounds",
        )
    ]

    prompt = render_consolidation_prompt("chapter_017", unresolved, registry)

    assert "Quote candidates to consolidate" in prompt
    assert "q001" in prompt
    assert "Mr. Pounds" in prompt
    assert "mr_john_pounds_adult" in prompt
    assert "Return JSON only" in prompt
    assert "Chapter text with marked quotes" not in prompt
