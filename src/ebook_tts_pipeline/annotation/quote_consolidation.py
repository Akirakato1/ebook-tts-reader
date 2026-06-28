from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from ebook_tts_pipeline.annotation.booknlp_candidates import QuoteAttributionCandidate
from ebook_tts_pipeline.annotation.quote_attribution import (
    QuoteAttributionResult,
    _registry_role_alias_candidates,
    validate_quote_attribution,
)
from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt
from ebook_tts_pipeline.registry import normalize_name


@dataclass(frozen=True)
class UnresolvedQuoteCandidate:
    quote_idx: int
    quote_id: str
    quote_text: str
    mention_phrase: str
    candidate_role_ids: List[str]


@dataclass(frozen=True)
class DeterministicConsolidationResult:
    resolved_quotes: Dict[int, str]
    unresolved: List[UnresolvedQuoteCandidate]


def consolidate_candidates_deterministically(
    candidates: List[QuoteAttributionCandidate],
    registry: Dict,
) -> DeterministicConsolidationResult:
    alias_candidates = _registry_role_alias_candidates(registry)
    resolved: Dict[int, str] = {}
    unresolved: List[UnresolvedQuoteCandidate] = []
    for candidate in candidates:
        possible = _candidate_role_matches(candidate, alias_candidates)
        if len(possible) == 1:
            resolved[candidate.quote_idx] = possible[0]
            continue
        unresolved.append(
            UnresolvedQuoteCandidate(
                quote_idx=candidate.quote_idx,
                quote_id=candidate.quote_id,
                quote_text=candidate.quote_text,
                mention_phrase=candidate.mention_phrase,
                candidate_role_ids=possible,
            )
        )
    return DeterministicConsolidationResult(resolved_quotes=resolved, unresolved=unresolved)


def _candidate_role_matches(
    candidate: QuoteAttributionCandidate,
    alias_candidates: Dict[str, set],
) -> List[str]:
    role_ids = set()
    for alias in [candidate.mention_phrase, *candidate.cluster_aliases]:
        role_ids.update(alias_candidates.get(normalize_name(alias), set()))
    return sorted(role_ids)


def render_consolidation_prompt(
    chapter: str,
    candidates: List[QuoteAttributionCandidate],
    registry: Dict,
) -> str:
    compact_candidates = [
        {
            "quote_idx": candidate.quote_idx,
            "quote_id": candidate.quote_id,
            "quote_text": candidate.quote_text,
            "booknlp_character_id": candidate.booknlp_character_id,
            "booknlp_mention_phrase": candidate.mention_phrase,
            "booknlp_cluster_aliases": list(candidate.cluster_aliases),
        }
        for candidate in candidates
    ]
    role_ids = [
        {
            **item,
            "role_id": role_id,
        }
        for role_id, item in _compact_registry_by_role_id(registry).items()
    ]
    return (
        "You consolidate local BookNLP quote-speaker candidates into exact audiobook registry roles.\n\n"
        f"Chapter: {chapter}\n\n"
        "Global registry role_ids are authoritative:\n"
        f"{json.dumps(role_ids, ensure_ascii=False, indent=2)}\n\n"
        "Quote candidates to consolidate:\n"
        f"{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- Use an exact global registry role_id when the BookNLP mention refers to a registry character.\n"
        "- If the speaker is not in the global registry, create a chapter-local local_speakers entry.\n"
        "- If the quote is not character dialogue, mark it narrator_quote.\n"
        "- Do not invent global role_ids.\n"
        "- Return JSON only using quote_attribution_v1 fields: roles, local_speakers, quotes.\n"
    )


def _compact_registry_by_role_id(registry: Dict) -> Dict[str, Dict]:
    compact = compact_registry_for_prompt(registry, include_aliases=True)
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return {}
    by_name = {
        normalize_name(str(item.get("name", ""))): item
        for item in compact
        if isinstance(item, dict)
    }
    output: Dict[str, Dict] = {}
    for role_id, record in characters.items():
        if not isinstance(record, dict):
            continue
        display_name = str(record.get("display_name") or role_id)
        output[str(record.get("role_id") or role_id)] = dict(by_name.get(normalize_name(display_name), {}))
    return output


class BookNlpSonnetConsolidationService:
    def __init__(self, client) -> None:
        self.client = client

    def consolidate(
        self,
        chapter: str,
        extraction,
        candidates: List[QuoteAttributionCandidate],
        registry: Dict,
    ) -> QuoteAttributionResult:
        deterministic = consolidate_candidates_deterministically(candidates, registry)
        if not deterministic.unresolved and len(deterministic.resolved_quotes) == len(extraction.quotes):
            result = _quote_result_from_resolved(extraction, deterministic.resolved_quotes)
            self._validate(result, extraction, registry)
            return result

        payload = self.client.complete_json(
            "Return valid JSON only.",
            render_consolidation_prompt(chapter, candidates, registry),
        )
        result = QuoteAttributionResult.from_dict(payload)
        self._validate(result, extraction, registry)
        return result

    def _validate(self, result: QuoteAttributionResult, extraction, registry: Dict) -> None:
        validate_quote_attribution(
            result,
            quote_indices=[quote.idx for quote in extraction.quotes],
            known_role_ids={str(role_id) for role_id in registry.get("characters", {})},
        )


def _quote_result_from_resolved(extraction, resolved_quotes: Dict[int, str]) -> QuoteAttributionResult:
    roles: List[str] = []
    quotes = []
    for quote in extraction.quotes:
        role_id = resolved_quotes[quote.idx]
        if role_id not in roles:
            roles.append(role_id)
        quotes.append((quote.idx, roles.index(role_id), "dialogue"))
    return QuoteAttributionResult(roles=roles, quotes=quotes)
