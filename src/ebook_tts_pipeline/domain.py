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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "source_path": self.source_path,
            "segmenter": self.segmenter,
            "sentences": [asdict(sentence) for sentence in self.sentences],
            "units": [unit.to_dict() for unit in self.annotation_units],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceArtifact":
        sentences = [Sentence.from_dict(item) for item in data["sentences"]]
        units = [
            SentenceUnit.from_dict(item)
            for item in data.get("units", [])
        ]
        if not units:
            units = [SentenceUnit(idx=sentence.idx, sentence_idx=sentence.idx, text=sentence.text) for sentence in sentences]
        return cls(
            chapter=str(data["chapter"]),
            source_path=str(data["source_path"]),
            segmenter=dict(data["segmenter"]),
            sentences=sentences,
            units=units,
        )

    @property
    def annotation_units(self) -> List[SentenceUnit]:
        if self.units:
            return self.units
        return [SentenceUnit(idx=sentence.idx, sentence_idx=sentence.idx, text=sentence.text) for sentence in self.sentences]


@dataclass(frozen=True)
class AnnotationResult:
    new_characters: List[Dict[str, Any]]
    roles: List[str]
    types: List[str]
    script: List[Tuple[int, int, int]]
    local_speakers: List[Dict[str, Any]] = field(default_factory=list)
    proposed_new_characters: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationResult":
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
            "new_characters": self.new_characters,
            "roles": self.roles,
            "types": self.types,
            "script": [list(row) for row in self.script],
        }
        if self.local_speakers:
            payload["local_speakers"] = self.local_speakers
        if self.proposed_new_characters:
            payload["proposed_new_characters"] = self.proposed_new_characters
        return payload
