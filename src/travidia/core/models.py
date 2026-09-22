"""Shared dataclasses for transcription results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """A single transcribed segment, optionally attributed to a speaker."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """The full result of transcribing an audio file."""

    segments: list[Segment]
    language: str
