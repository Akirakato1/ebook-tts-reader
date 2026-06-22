from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sentence":
        return cls(idx=int(data["idx"]), text=str(data["text"]))


@dataclass(frozen=True)
class SentenceUnit:
    idx: int
    sentence_idx: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceUnit":
        return cls(
            idx=int(data["idx"]),
            sentence_idx=int(data["sentence_idx"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class SentenceArtifact:
    chapter: str
    source_path: str
    segmenter: Dict[str, str]
    sentences: List[Sentence]
    units: List[SentenceUnit] = field(default_factory=list)
    source_sentences: List[Sentence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "chapter": self.chapter,
            "source_path": self.source_path,
            "segmenter": self.segmenter,
            "sentences": [asdict(sentence) for sentence in self.sentences],
            "units": [unit.to_dict() for unit in self.annotation_units],
        }
        if self.source_sentences:
            payload["source_sentences"] = [asdict(sentence) for sentence in self.source_sentences]
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceArtifact":
        serialized_sentences = [Sentence.from_dict(item) for item in data["sentences"]]
        units = [
            SentenceUnit.from_dict(item)
            for item in data.get("units", [])
        ]
        if not units:
            units = [
                SentenceUnit(idx=sentence.idx, sentence_idx=sentence.idx, text=sentence.text)
                for sentence in serialized_sentences
            ]
        source_sentences = [
            Sentence.from_dict(item)
            for item in data.get("source_sentences", [])
        ]
        if not source_sentences:
            source_sentences = serialized_sentences
        sentences = [
            Sentence(idx=unit.idx, text=unit.text)
            for unit in units
        ]
        return cls(
            chapter=str(data["chapter"]),
            source_path=str(data["source_path"]),
            segmenter=dict(data["segmenter"]),
            sentences=sentences,
            units=units,
            source_sentences=source_sentences,
        )

    @property
    def annotation_units(self) -> List[SentenceUnit]:
        if self.units:
            return self.units
        return [SentenceUnit(idx=sentence.idx, sentence_idx=sentence.idx, text=sentence.text) for sentence in self.sentences]


@dataclass(frozen=True)
class AnnotationResult:
    roles: List[str]
    types: List[str]
    script: List[Tuple[int, int, int]]
    new_characters: List[Dict[str, Any]] = field(default_factory=list)
    local_speakers: List[Dict[str, Any]] = field(default_factory=list)
    proposed_new_characters: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationResult":
        if _is_compact_quote_annotation(data):
            types, script = _quote_annotation_types_and_script(data)
            return cls(
                new_characters=list(data.get("new_characters", [])),
                roles=[str(role) for role in data["roles"]],
                types=types,
                script=script,
                local_speakers=list(data.get("local_speakers", [])),
                proposed_new_characters=list(data.get("proposed_new_characters", [])),
            )
        return cls(
            new_characters=list(data.get("new_characters", [])),
            roles=[str(role) for role in data["roles"]],
            types=[str(item) for item in data["types"]],
            script=[tuple(int(value) for value in row) for row in data["script"]],
            local_speakers=list(data.get("local_speakers", [])),
            proposed_new_characters=list(data.get("proposed_new_characters", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "roles": self.roles,
            "types": self.types,
            "script": [list(row) for row in self.script],
        }
        if self.new_characters:
            payload["new_characters"] = self.new_characters
        if self.local_speakers:
            payload["local_speakers"] = self.local_speakers
        if self.proposed_new_characters:
            payload["proposed_new_characters"] = self.proposed_new_characters
        return payload


def _is_compact_quote_annotation(data: Dict[str, Any]) -> bool:
    return "quotes" in data and ("types" not in data or "script" not in data)


def _quote_annotation_types_and_script(data: Dict[str, Any]) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    types: List[str] = []
    script: List[Tuple[int, int, int]] = []
    for row in data.get("quotes", []):
        values = list(row)
        if len(values) < 2:
            continue
        quote_idx = int(values[0])
        role_idx = int(values[1])
        quote_type = str(values[2]) if len(values) > 2 else "dialogue"
        if quote_type not in types:
            types.append(quote_type)
        script.append((role_idx, types.index(quote_type), quote_idx))
    if not types:
        types = ["dialogue"]
    return types, script
