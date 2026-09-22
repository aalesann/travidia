"""Tests for the shared transcription result dataclasses."""

from travidia.core.models import Segment, TranscriptionResult


def test_segment_defaults_speaker_to_none():
    segment = Segment(start=0.0, end=1.5, text="hello world")

    assert segment.start == 0.0
    assert segment.end == 1.5
    assert segment.text == "hello world"
    assert segment.speaker is None


def test_segment_accepts_speaker():
    segment = Segment(start=0.0, end=1.5, text="hello", speaker="SPEAKER_00")

    assert segment.speaker == "SPEAKER_00"


def test_transcription_result_holds_segments_and_language():
    segments = [
        Segment(start=0.0, end=1.0, text="hi"),
        Segment(start=1.0, end=2.0, text="there", speaker="SPEAKER_01"),
    ]

    result = TranscriptionResult(segments=segments, language="en")

    assert result.segments == segments
    assert result.language == "en"
