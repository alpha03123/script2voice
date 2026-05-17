from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptBlock:
    index: int
    tag: str
    text: str
    visual_notes: list[str]


@dataclass(frozen=True)
class SentenceAudio:
    text: str
    duration: float
    start: float = 0.0
    end: float = 0.0


@dataclass(frozen=True)
class OutputBlock:
    index: int
    tag: str
    text: str
    visual_notes: list[str]
    wav: str
    srt: str
    duration: float
    sample_rate: int
    sentences: list[SentenceAudio]
