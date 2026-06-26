from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

from ebook_tts_pipeline.annotation.anthropic_client import AnnotationModelOutputError
from ebook_tts_pipeline.annotation.prompts import SYSTEM_PROMPT
from ebook_tts_pipeline.annotation.quotes import QuoteExtraction
from ebook_tts_pipeline.annotation.registry_summary import compact_registry_for_prompt
from ebook_tts_pipeline.debug_logging import FailureLogger
from ebook_tts_pipeline.registry import normalize_name


ALLOWED_QUOTE_TYPES = {"dialogue", "narrator_quote"}


class QuoteAttributionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class QuoteAttributionResult:
    roles: List[str]
    quotes: List[Tuple[int, int, str]]
    local_speakers: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuoteAttributionResult":
        roles = [str(role) for role in data["roles"]]
        return cls(
            roles=roles,
            local_speakers=list(data.get("local_speakers", [])),
            quotes=[
                _normalize_quote_row(row, roles)
                for row in data["quotes"]
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "roles": self.roles,
            "quotes": [_compact_quote_row(row) for row in self.quotes],
        }
        if self.local_speakers:
            payload["local_speakers"] = self.local_speakers
        return payload


def render_quote_attribution_prompt(
    chapter: str,
    extraction: QuoteExtraction,
    registry: Dict[str, Any],
) -> str:
    compact_registry = _compact_registry_with_role_ids(registry)
    quote_ids = [quote.quote_id for quote in extraction.quotes]
    return (
        "You are attributing quoted dialogue in a novel chapter for audiobook generation.\n\n"
        f"Chapter: {chapter}\n\n"
        "Global recurring characters. Existing registry roles are authoritative:\n"
        f"{json.dumps(compact_registry, ensure_ascii=False, indent=2)}\n\n"
        "Chapter text with marked quotes:\n"
        f"{extraction.to_marked_text()}\n\n"
        f"Quote IDs to attribute: {json.dumps(quote_ids)}\n\n"
        "Required workflow:\n"
        "Step 1: Review the global registry. If a quote speaker is present there, use that exact global role_id.\n"
        "Step 2: Build the chapter-local registry. local_speakers is the local registry for this chapter only: "
        "include useful chapter-only speakers who are not present in the global registry, especially unnamed roles "
        "such as guards, clerks, nurses, officers, children, or other one-chapter speakers.\n"
        "Step 3: Build roles from the union of global registry role_ids and chapter-local local_ids. "
        "roles must contain only global role_id values and local_speakers.local_id values.\n"
        "Step 4: Assign every quote exactly once by quote_idx and role_idx, where role_idx points into roles.\n\n"
        "Rules:\n"
        "- Attribute every marked quote exactly once.\n"
        "- Choose a global role_id when the speaker is a recurring registry character.\n"
        "- If the same person has multiple age stages in the registry, choose the active age-stage role_id.\n"
        "- If the speaker is not in the global registry and appears chapter-only, create a local speaker.\n"
        "- Do not reuse one local speaker for distinct unnamed people. Different scenes, occupations, descriptions, or speech contexts require separate local_speakers.\n"
        "- Every local_speakers entry must be assigned to at least one quote. Do not create unused local speakers.\n"
        "- Use the exact local_id in roles for local speakers. The label is display-only and must not be used as a role_id.\n"
        "- Prefer descriptive local_id slugs for chapter-only speakers when possible, such as security_guard_001 or waitress_001, instead of generic local_001.\n"
        "- Use local_speakers only for real character dialogue by an agentic speaker who should get a distinct voice, "
        "such as an unnamed guard, clerk, nurse, officer, child, or caller. Do not create local_speakers for "
        "non-character functional quoted text such as automated phone-system messages, signs, titles, labels, "
        "quoted terms, written snippets, or ambient/system recordings. Mark those rows as narrator_quote instead.\n"
        "- Do not create global registry characters in this output.\n"
        "- Do not label normal quoted dialogue as Narrator.\n"
        "- Use narrator_quote only when quote marks are not spoken dialogue, such as titles, quoted terms, or sarcasm.\n"
        "- narrator_quote is a quote type only. It is never a role_id, speaker, or local_id.\n"
        "- Return JSON only. Do not include quote text or explanations.\n\n"
        "Output schema:\n"
        "{\n"
        '  "roles": ["role_id_or_exact_local_id"],\n'
        '  "local_speakers": [\n'
        "    {\n"
        '      "local_id": "security_guard_001",\n'
        '      "label": "Security Guard",\n'
        '      "profile": {\n'
        '        "age_stage": "adult|child|teen|elder|unknown",\n'
        '        "gender": "female|male|nonbinary|unknown",\n'
        '        "race_or_ethnicity": null,\n'
        '        "accent": null,\n'
        '        "occupation": null,\n'
        '        "personality": ["short trait"]\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "quotes": [[1, 0]]\n'
        "}\n"
        "In quotes rows, use numeric quote_idx and numeric role_idx: q001 is quote_idx 1, "
        "and role_idx is the zero-based index into roles. Omit the third item for normal dialogue; "
        'only add a third item, "narrator_quote", for quoted terms, titles, or other marked text '
        "that should be read by the narrator rather than spoken by a character. Before returning, "
        "check that every local_speakers.local_id appears in roles and is assigned to at least one quote, "
        "and that no local speaker label appears in roles unless the label is identical to local_id.\n"
    )


def render_quote_attribution_repair_prompt(
    original_prompt: str,
    invalid_payload: Any,
    validation_error: str,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed validation and must be corrected.\n"
        f"Validation error: {validation_error}\n\n"
        "Previous invalid JSON:\n"
        f"{json.dumps(invalid_payload, ensure_ascii=False)}\n\n"
        f"{_render_repair_diagnostics(diagnostics or [])}"
        "Return corrected JSON only using the same output schema. "
        "Do not add explanations. If a quote row uses the third item \"narrator_quote\", "
        "do not create a local_speakers entry for non-character functional quoted text such as "
        "automated phone-system messages, signs, labels, written snippets, or ambient/system recordings. "
        "Every local speaker role must have a matching local_speakers profile, and every "
        "local_speakers entry must be assigned to at least one quote.\n"
    )


def render_quote_attribution_non_json_repair_prompt(
    original_prompt: str,
    invalid_text: str,
    parse_error: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous response was not valid JSON and could not be parsed.\n"
        f"Parse error: {parse_error}\n\n"
        "Previous invalid response text:\n"
        f"{json.dumps(_bounded_prompt_text(invalid_text), ensure_ascii=False)}\n\n"
        "Return corrected JSON only using the same output schema. "
        "Do not include reasoning, Markdown headings, bullet points, or explanations. "
        "The response must begin with { and end with }.\n"
    )


def validate_quote_attribution(
    result: QuoteAttributionResult,
    quote_indices: Iterable[int],
    known_role_ids: Set[str],
) -> None:
    errors: List[str] = []
    expected = set(int(index) for index in quote_indices)
    seen: Set[int] = set()
    duplicate: Set[int] = set()
    local_ids = _validate_local_speakers(result.local_speakers, errors)
    local_id_labels = {
        normalize_name(str(speaker.get("local_id", ""))): str(speaker.get("local_id", "")).strip()
        for speaker in result.local_speakers
        if str(speaker.get("local_id", "")).strip()
    }
    used_local_ids: Set[str] = set()
    normalized_known = {normalize_name(role_id) for role_id in known_role_ids}

    for quote_idx, role_idx, quote_type in result.quotes:
        if quote_idx in seen:
            duplicate.add(quote_idx)
        seen.add(quote_idx)
        if role_idx < 0 or role_idx >= len(result.roles):
            errors.append(f"role index out of range for quote {quote_idx}: {role_idx}")
            continue
        if quote_type not in ALLOWED_QUOTE_TYPES:
            errors.append(f"invalid quote type for quote {quote_idx}: {quote_type}")
            continue
        if quote_type == "narrator_quote":
            continue
        role = result.roles[role_idx]
        normalized_role = normalize_name(role)
        if normalized_role in local_ids:
            used_local_ids.add(normalized_role)
        if quote_type == "dialogue" and normalized_role == normalize_name("Narrator"):
            errors.append(f"Narrator cannot speak dialogue quote {quote_idx}")
        if (
            normalized_role != normalize_name("Narrator")
            and normalized_role not in normalized_known
            and normalized_role not in local_ids
        ):
            errors.append(f"local role missing profile for quote {quote_idx}: {role}")

    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append(f"missing quote assignments: {missing}")
    if extra:
        errors.append(f"unknown quote assignments: {extra}")
    if duplicate:
        errors.append(f"duplicate quote assignments: {sorted(duplicate)}")
    unused_local_ids = [
        local_id_labels.get(local_id, local_id)
        for local_id in sorted(local_ids - used_local_ids)
    ]
    if unused_local_ids:
        errors.append(f"unused local speakers: {', '.join(unused_local_ids)}")

    if errors:
        raise QuoteAttributionValidationError("; ".join(errors))


class QuoteAttributionService:
    def __init__(
        self,
        client: Any,
        failure_logger: FailureLogger | None = None,
        repair_retries: int = 1,
    ) -> None:
        self.client = client
        self.failure_logger = failure_logger
        self.repair_retries = max(0, int(repair_retries))

    def attribute_quotes(
        self,
        chapter: str,
        extraction: QuoteExtraction,
        registry: Dict[str, Any],
    ) -> QuoteAttributionResult:
        prompt = render_quote_attribution_prompt(chapter, extraction, registry)
        quote_ids = [quote.quote_id for quote in extraction.quotes]
        quote_indices = [quote.idx for quote in extraction.quotes]
        known_role_ids = set(_registry_role_ids(registry))
        payload, prompt, repairs_used = self._complete_initial_payload(
            chapter=chapter,
            prompt=prompt,
            quote_ids=quote_ids,
        )

        for attempt in range(repairs_used, self.repair_retries + 1):
            try:
                result = _canonicalize_quote_attribution_result(
                    QuoteAttributionResult.from_dict(payload),
                    registry=registry,
                )
                validate_quote_attribution(
                    result,
                    quote_indices=quote_indices,
                    known_role_ids=known_role_ids,
                )
                return result
            except Exception as exc:
                self._log_failure(
                    "quote_attribution_validation_failed",
                    chapter=chapter,
                    prompt=prompt,
                    quote_ids=quote_ids,
                    payload=payload,
                    exc=exc,
                    details={
                        "attempt": attempt,
                        "repair_available": attempt < self.repair_retries,
                    },
                )
                if attempt >= self.repair_retries:
                    raise
                prompt = render_quote_attribution_repair_prompt(
                    prompt,
                    payload,
                    str(exc),
                    diagnostics=_repair_diagnostics_for_payload(payload, extraction, registry),
                )
                payload = self._complete_json(
                    chapter=chapter,
                    prompt=prompt,
                    quote_ids=quote_ids,
                    call_type="repair",
                )

        raise RuntimeError("unreachable quote attribution repair state")

    def _complete_initial_payload(
        self,
        chapter: str,
        prompt: str,
        quote_ids: List[str],
    ) -> Tuple[Any, str, int]:
        try:
            return (
                self._complete_json(
                    chapter=chapter,
                    prompt=prompt,
                    quote_ids=quote_ids,
                    call_type="attribution",
                ),
                prompt,
                0,
            )
        except AnnotationModelOutputError as exc:
            raw_text = getattr(exc, "raw_text", None)
            if self.repair_retries <= 0 or not raw_text:
                raise
            repair_prompt = render_quote_attribution_non_json_repair_prompt(prompt, raw_text, str(exc))
            return (
                self._complete_json(
                    chapter=chapter,
                    prompt=repair_prompt,
                    quote_ids=quote_ids,
                    call_type="repair_non_json",
                ),
                repair_prompt,
                1,
            )

    def _complete_json(
        self,
        chapter: str,
        prompt: str,
        quote_ids: List[str],
        call_type: str,
    ) -> Any:
        try:
            return self.client.complete_json(SYSTEM_PROMPT, prompt)
        except Exception as exc:
            self._log_failure(
                "quote_attribution_model_failed",
                chapter=chapter,
                prompt=prompt,
                quote_ids=quote_ids,
                payload=None,
                exc=exc,
                details={"call_type": call_type},
            )
            raise

    def _log_failure(
        self,
        event_type: str,
        chapter: str,
        prompt: str,
        quote_ids: List[str],
        payload: Any,
        exc: BaseException,
        details: Dict[str, Any] | None = None,
    ) -> None:
        if self.failure_logger is None:
            return
        event_details = {
            "chapter": chapter,
            "quote_ids": quote_ids,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt,
            "payload": payload,
            "raw_model_text": getattr(exc, "raw_text", None),
        }
        if details:
            event_details.update(details)
        self.failure_logger.with_context(chapter=chapter).write_failure(event_type, event_details, exc=exc)


def _compact_registry_with_role_ids(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = compact_registry_for_prompt(registry, include_aliases=True)
    role_ids = list(_registry_role_ids(registry))
    for index, record in enumerate(compact):
        if index < len(role_ids):
            record["role_id"] = role_ids[index]
    return compact


def _bounded_prompt_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n\n[...truncated failed model response...]\n\n{tail}"


def _registry_role_ids(registry: Dict[str, Any]) -> List[str]:
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return []
    return [str(record.get("role_id") or role_id) for role_id, record in characters.items() if isinstance(record, dict)]


def _repair_diagnostics_for_payload(
    payload: Any,
    extraction: QuoteExtraction,
    registry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        result = _canonicalize_quote_attribution_result(
            QuoteAttributionResult.from_dict(payload),
            registry=registry,
        )
    except Exception:
        return []

    known_role_ids = {normalize_name(role_id) for role_id in _registry_role_ids(registry)}
    local_ids = {
        normalize_name(str(speaker.get("local_id", "")))
        for speaker in result.local_speakers
        if str(speaker.get("local_id", "")).strip()
    }
    quote_by_idx = {quote.idx: quote for quote in extraction.quotes}
    alias_candidates = _registry_role_alias_candidates(registry)
    by_role: Dict[str, Dict[str, Any]] = {}

    for quote_idx, role_idx, quote_type in result.quotes:
        if quote_type == "narrator_quote" or role_idx < 0 or role_idx >= len(result.roles):
            continue
        role = result.roles[role_idx]
        normalized_role = normalize_name(role)
        if (
            normalized_role == normalize_name("Narrator")
            or normalized_role in known_role_ids
            or normalized_role in local_ids
        ):
            continue
        entry = by_role.setdefault(
            role,
            {
                "role": role,
                "quotes": [],
                "candidates": [
                    _registry_candidate_summary(registry, candidate_role_id)
                    for candidate_role_id in sorted(alias_candidates.get(normalized_role, set()))
                ],
            },
        )
        quote = quote_by_idx.get(quote_idx)
        entry["quotes"].append(
            {
                "quote_idx": quote_idx,
                "quote_id": f"q{quote_idx:03d}",
                "text": _bounded_quote_text(quote.text if quote is not None else ""),
            }
        )

    return list(by_role.values())


def _render_repair_diagnostics(diagnostics: List[Dict[str, Any]]) -> str:
    if not diagnostics:
        return ""
    lines = [
        "Invalid role diagnostics:\n",
        "For each invalid role below, consolidate to an exact global registry role_id when the speaker is a registry character. "
        "If the speaker is not represented by the global registry, add a local_speakers entry and make roles reference that local_id. "
        "Do not invent new global role_ids.\n",
    ]
    for diagnostic in diagnostics:
        lines.append(f"- invalid role: {diagnostic['role']}")
        lines.append("  error: role is neither an exact global registry role_id nor a local_speakers.local_id.")
        quotes = diagnostic.get("quotes", [])
        if quotes:
            lines.append("  quotes using this invalid role:")
            for quote in quotes:
                lines.append(f"    - {quote['quote_id']}: {json.dumps(quote['text'], ensure_ascii=False)}")
        candidates = diagnostic.get("candidates", [])
        if candidates:
            lines.append("  registry candidates from alias matching:")
            for candidate in candidates:
                lines.append(
                    "    - "
                    f"role_id: {candidate['role_id']}; "
                    f"display_name: {json.dumps(candidate['display_name'], ensure_ascii=False)}; "
                    f"age_stage: {json.dumps(candidate['age_stage'], ensure_ascii=False)}; "
                    f"aliases: {json.dumps(candidate['aliases'], ensure_ascii=False)}"
                )
            lines.append(
                "  repair: if chapter context matches one of these candidates, replace it with the exact registry role_id."
            )
        else:
            lines.append("  registry candidates from alias matching: none.")
        lines.append("  repair: if no registry candidate fits, add a local_speakers entry and use that local_id in roles.")
    return "\n".join(lines) + "\n\n"


def _bounded_quote_text(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _registry_candidate_summary(registry: Dict[str, Any], role_id: str) -> Dict[str, Any]:
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return {"role_id": role_id, "display_name": "", "age_stage": "", "aliases": []}
    for key, record in characters.items():
        if not isinstance(record, dict):
            continue
        canonical = str(record.get("role_id") or key).strip()
        if canonical != role_id:
            continue
        identity = record.get("identity_profile", {})
        identity = identity if isinstance(identity, dict) else {}
        return {
            "role_id": canonical,
            "display_name": str(record.get("display_name") or "").strip(),
            "age_stage": str(record.get("age_stage") or identity.get("age_stage") or "").strip(),
            "aliases": [str(alias) for alias in record.get("aliases", []) if str(alias).strip()],
        }
    return {"role_id": role_id, "display_name": "", "age_stage": "", "aliases": []}


def _canonicalize_quote_attribution_result(
    result: QuoteAttributionResult,
    registry: Dict[str, Any],
) -> QuoteAttributionResult:
    roles = list(result.roles)
    local_ids = {
        normalize_name(str(speaker.get("local_id", "")))
        for speaker in result.local_speakers
        if str(speaker.get("local_id", "")).strip()
    }
    registry_aliases = _unique_registry_role_aliases(registry)
    for index, role in enumerate(roles):
        normalized_role = normalize_name(role)
        if normalized_role == normalize_name("narrator_quote"):
            roles[index] = "Narrator"
            continue
        if normalized_role in local_ids:
            continue
        canonical_role = registry_aliases.get(normalized_role)
        if canonical_role:
            roles[index] = canonical_role

    dialogue_local_ids: Set[str] = set()
    narrator_quote_local_ids: Set[str] = set()
    for _quote_idx, role_idx, quote_type in result.quotes:
        if role_idx < 0 or role_idx >= len(roles):
            continue
        normalized_role = normalize_name(roles[role_idx])
        if normalized_role not in local_ids:
            continue
        if quote_type == "narrator_quote":
            narrator_quote_local_ids.add(normalized_role)
        else:
            dialogue_local_ids.add(normalized_role)

    redundant_narrator_quote_local_ids = narrator_quote_local_ids - dialogue_local_ids
    local_speakers = [
        speaker
        for speaker in result.local_speakers
        if normalize_name(str(speaker.get("local_id", ""))) not in redundant_narrator_quote_local_ids
    ]

    narrator_index = -1
    quotes = []
    for quote_idx, role_idx, quote_type in result.quotes:
        if quote_type == "narrator_quote":
            if narrator_index < 0:
                narrator_index = _ensure_role_index(roles, "Narrator")
            quotes.append((quote_idx, narrator_index, quote_type))
        else:
            quotes.append((quote_idx, role_idx, quote_type))
    roles, quotes = _prune_unreferenced_roles(roles, quotes)
    return QuoteAttributionResult(roles=roles, quotes=quotes, local_speakers=local_speakers)


def _unique_registry_role_aliases(registry: Dict[str, Any]) -> Dict[str, str]:
    candidates = _registry_role_alias_candidates(registry)
    return {
        alias: next(iter(role_ids))
        for alias, role_ids in candidates.items()
        if len(role_ids) == 1
    }


def _registry_role_alias_candidates(registry: Dict[str, Any]) -> Dict[str, Set[str]]:
    candidates: Dict[str, Set[str]] = {}
    characters = registry.get("characters", {})
    if not isinstance(characters, dict):
        return {}
    for role_id, record in characters.items():
        if not isinstance(record, dict):
            continue
        canonical = str(record.get("role_id") or role_id).strip()
        if not canonical:
            continue
        for alias in _registry_record_aliases(canonical, record):
            normalized = normalize_name(alias)
            if normalized:
                candidates.setdefault(normalized, set()).add(canonical)
    return candidates


def _registry_record_aliases(canonical: str, record: Dict[str, Any]) -> Set[str]:
    identity = record.get("identity_profile", {})
    identity = identity if isinstance(identity, dict) else {}
    age_stage = str(record.get("age_stage") or identity.get("age_stage") or "").strip()
    display_name = str(record.get("display_name") or "").strip()
    aliases = {canonical, canonical.replace("_", " ")}
    for value in record.get("aliases", []):
        text = str(value).strip()
        if text:
            aliases.add(text)
            if age_stage:
                aliases.add(f"{text} {age_stage}")
    if display_name:
        aliases.add(display_name)
        if age_stage:
            aliases.add(f"{display_name} {age_stage}")
        aliases.update(_short_honorific_aliases(display_name, age_stage))
    return aliases


def _short_honorific_aliases(display_name: str, age_stage: str) -> Set[str]:
    tokens = display_name.split()
    if len(tokens) < 3:
        return set()
    honorifics = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord"}
    honorific = tokens[0].rstrip(".").lower()
    if honorific not in honorifics:
        return set()
    short = f"{tokens[0]} {tokens[-1]}"
    aliases = {short}
    if age_stage:
        aliases.add(f"{short} {age_stage}")
    return aliases


def _prune_unreferenced_roles(
    roles: List[str],
    quotes: List[Tuple[int, int, str]],
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    used_indices = {role_idx for _quote_idx, role_idx, _quote_type in quotes}
    old_to_new: Dict[int, int] = {}
    pruned_roles: List[str] = []
    for old_index, role in enumerate(roles):
        if old_index not in used_indices:
            continue
        old_to_new[old_index] = len(pruned_roles)
        pruned_roles.append(role)
    return pruned_roles, [
        (quote_idx, old_to_new[role_idx], quote_type)
        for quote_idx, role_idx, quote_type in quotes
    ]


def _ensure_role_index(roles: List[str], role: str) -> int:
    normalized = normalize_name(role)
    for index, candidate in enumerate(roles):
        if normalize_name(candidate) == normalized:
            return index
    roles.append(role)
    return len(roles) - 1


def _compact_quote_row(row: Tuple[int, int, str]) -> List[Any]:
    quote_idx, role_idx, quote_type = row
    if quote_type == "dialogue":
        return [quote_idx, role_idx]
    return [quote_idx, role_idx, quote_type]


def _normalize_quote_row(row: Any, roles: List[str]) -> Tuple[int, int, str]:
    if isinstance(row, dict):
        quote_ref = _first_present(row.get("quote_idx"), row.get("quote_id"), row.get("quote"))
        role_ref = _first_present(row.get("role_idx"), row.get("role_id"), row.get("role"), row.get("speaker"))
        quote_type = str(_first_present(row.get("type"), row.get("quote_type"), "dialogue"))
        return (_quote_index(quote_ref), _role_index(role_ref, roles), quote_type)

    values = list(row) if isinstance(row, (list, tuple)) else []
    if len(values) < 2:
        raise ValueError(f"Quote attribution row must have at least two values: {row!r}")
    quote_type = _quote_type_from_values(values[2:])
    first, second = values[0], values[1]
    if _looks_like_quote_ref(second) and not _looks_like_quote_ref(first):
        role_ref, quote_ref = first, second
    else:
        quote_ref, role_ref = first, second
    return (_quote_index(quote_ref), _role_index(role_ref, roles), quote_type)


def _quote_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"q0*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid quote reference: {value!r}")


def _role_index(value: Any, roles: List[str]) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    normalized = normalize_name(text)
    for index, role in enumerate(roles):
        if normalize_name(role) == normalized:
            return index
    raise ValueError(f"Invalid role reference: {value!r}")


def _looks_like_quote_ref(value: Any) -> bool:
    if isinstance(value, int):
        return True
    text = str(value).strip()
    return text.isdigit() or re.fullmatch(r"q0*\d+", text, flags=re.IGNORECASE) is not None


def _quote_type_from_values(values: List[Any]) -> str:
    for value in values:
        text = str(value).strip()
        if text in ALLOWED_QUOTE_TYPES:
            return text
    return "dialogue"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _validate_local_speakers(
    local_speakers: List[Dict[str, Any]],
    errors: List[str],
) -> Set[str]:
    local_ids: Set[str] = set()
    for speaker in local_speakers:
        local_id = str(speaker.get("local_id", "")).strip()
        profile = speaker.get("profile")
        if not local_id:
            errors.append("local speaker missing local_id")
            continue
        if not isinstance(profile, dict):
            errors.append(f"local speaker missing profile: {local_id}")
            continue
        local_ids.add(normalize_name(local_id))
    return local_ids
