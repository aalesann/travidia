"""Tests for txt/srt/json export formats."""

import json
from pathlib import Path

from travidia.core.export import write_json, write_srt, write_txt
from travidia.core.models import Segment, TranscriptionResult


def make_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(start=0.0, end=1.5, text="Hello there.", speaker="SPEAKER_00"),
            Segment(start=1.5, end=3.256, text="General Kenobi.", speaker="SPEAKER_01"),
            Segment(start=3.256, end=5.0, text="No speaker here."),
        ],
        language="en",
    )


def test_write_txt_prefixes_speaker_when_present(tmp_path: Path):
    output_path = tmp_path / "out.txt"

    write_txt(make_result(), output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[SPEAKER_00] Hello there."
    assert lines[1] == "[SPEAKER_01] General Kenobi."
    assert lines[2] == "No speaker here."


def test_write_srt_formats_timestamps_and_numbering(tmp_path: Path):
    output_path = tmp_path / "out.srt"

    write_srt(make_result(), output_path)

    content = output_path.read_text(encoding="utf-8")
    blocks = content.strip().split("\n\n")
    assert len(blocks) == 3

    first_lines = blocks[0].splitlines()
    assert first_lines[0] == "1"
    assert first_lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert first_lines[2] == "Hello there."

    second_lines = blocks[1].splitlines()
    assert second_lines[0] == "2"
    assert second_lines[1] == "00:00:01,500 --> 00:00:03,256"
    assert second_lines[2] == "General Kenobi."

    third_lines = blocks[2].splitlines()
    assert third_lines[0] == "3"
    assert third_lines[1] == "00:00:03,256 --> 00:00:05,000"


def test_write_srt_handles_hour_boundary(tmp_path: Path):
    output_path = tmp_path / "out.srt"
    result = TranscriptionResult(
        segments=[Segment(start=3661.789, end=3665.0, text="An hour in.")],
        language="en",
    )

    write_srt(result, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "01:01:01,789 --> 01:01:05,000" in content


def test_write_json_serializes_segments_and_language(tmp_path: Path):
    output_path = tmp_path / "out.json"

    write_json(make_result(), output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert len(data["segments"]) == 3
    assert data["segments"][0] == {
        "start": 0.0,
        "end": 1.5,
        "text": "Hello there.",
        "speaker": "SPEAKER_00",
    }
    assert data["segments"][2]["speaker"] is None
