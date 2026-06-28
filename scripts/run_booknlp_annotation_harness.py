from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-root", required=True)
    parser.add_argument("--chapters", nargs="+", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = build_harness_report(
        book_slug=Path(args.book_root).name,
        chapters=args.chapters,
        deterministic_quotes=0,
        sonnet_quotes=0,
        failed_quotes=0,
        sonnet_prompt_chars=0,
        old_full_prompt_chars=0,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
