from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from ebook_tts_pipeline.annotation.anthropic_client import AnthropicJsonClient
from ebook_tts_pipeline.annotation.booknlp_artifacts import (
    character_aliases_from_entities,
    parse_booknlp_entities,
    parse_booknlp_quotes,
)
from ebook_tts_pipeline.annotation.booknlp_candidates import map_booknlp_quotes_to_extraction
from ebook_tts_pipeline.annotation.quote_consolidation import (
    BookNlpSonnetConsolidationService,
    consolidate_candidates_deterministically,
    render_consolidation_prompt,
)
from ebook_tts_pipeline.annotation.quotes import extract_quoted_dialogue
from ebook_tts_pipeline.config import PipelineConfig
from ebook_tts_pipeline.json_io import read_json, write_json_atomic
from ebook_tts_pipeline.paths import BookPaths


def build_harness_report(
    book_slug: str,
    chapters: List[str],
    deterministic_quotes: int,
    sonnet_quotes: int,
    failed_quotes: int,
    sonnet_prompt_chars: int,
    old_full_prompt_chars: int,
) -> Dict:
    return {
        "book_slug": book_slug,
        "chapters": list(chapters),
        "deterministic_quotes": deterministic_quotes,
        "sonnet_quotes": sonnet_quotes,
        "failed_quotes": failed_quotes,
        "sonnet_prompt_chars": sonnet_prompt_chars,
        "old_full_prompt_chars": old_full_prompt_chars,
        "estimated_prompt_char_savings": max(0, old_full_prompt_chars - sonnet_prompt_chars),
    }


def run_cached_harness(
    paths: BookPaths,
    chapters: List[str],
    client,
    write_production_annotations: bool = False,
) -> Dict:
    registry = read_json(paths.registry) if paths.registry.exists() else {}
    quote_rows = parse_booknlp_quotes(paths.booknlp_output_dir / "book.quotes")
    cluster_aliases = _load_cluster_aliases(paths)
    service = BookNlpSonnetConsolidationService(client)
    deterministic_quotes = 0
    sonnet_quotes = 0
    failed_quotes = 0
    sonnet_prompt_chars = 0
    old_full_prompt_chars = 0
    sidecar_root = paths.booknlp_dir / "harness_annotations"

    for chapter in chapters:
        chapter_text = paths.chapter_text(chapter).read_text(encoding="utf-8", errors="replace")
        extraction = extract_quoted_dialogue(chapter_text)
        old_full_prompt_chars += len(chapter_text)
        candidates = map_booknlp_quotes_to_extraction(
            chapter,
            extraction,
            quote_rows,
            cluster_aliases=cluster_aliases,
        )
        deterministic = consolidate_candidates_deterministically(candidates, registry)
        deterministic_quotes += len(deterministic.resolved_quotes)
        unresolved_count = max(0, len(extraction.quotes) - len(deterministic.resolved_quotes))
        if unresolved_count:
            sonnet_quotes += unresolved_count
            sonnet_prompt_chars += len(render_consolidation_prompt(chapter, candidates, registry))
        try:
            result = service.consolidate(chapter, extraction, candidates, registry)
        except Exception:
            failed_quotes += len(extraction.quotes)
            raise
        target = paths.annotation(chapter) if write_production_annotations else sidecar_root / f"{chapter}.annotation.json"
        write_json_atomic(target, result.to_dict())

    return build_harness_report(
        book_slug=paths.root.name,
        chapters=chapters,
        deterministic_quotes=deterministic_quotes,
        sonnet_quotes=sonnet_quotes,
        failed_quotes=failed_quotes,
        sonnet_prompt_chars=sonnet_prompt_chars,
        old_full_prompt_chars=old_full_prompt_chars,
    )


def _load_cluster_aliases(paths: BookPaths) -> Dict[str, List[str]]:
    entities_path = paths.booknlp_output_dir / "book.entities"
    if not entities_path.exists():
        return {}
    return character_aliases_from_entities(parse_booknlp_entities(entities_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapters", nargs="+", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--write-production-annotations", action="store_true")
    args = parser.parse_args()
    paths = BookPaths(Path(args.book_root))
    config = PipelineConfig.from_env(str(paths.root))
    client = _build_client(config)
    report = run_cached_harness(
        paths,
        args.chapters,
        client=client,
        write_production_annotations=bool(args.write_production_annotations),
    )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _build_client(config: PipelineConfig):
    if not config.anthropic_api_key:
        return _MissingSonnetClient()
    return AnthropicJsonClient(
        api_key=config.anthropic_api_key,
        model=config.anthropic_model,
        temperature=config.anthropic_temperature,
        max_tokens=config.anthropic_max_tokens,
    )


class _MissingSonnetClient:
    def complete_json(self, system_prompt: str, user_prompt: str):
        raise RuntimeError("ANTHROPIC_API_KEY is required when BookNLP candidates need Sonnet consolidation.")


if __name__ == "__main__":
    main()
