"""Export a TranscriptionResult as plain text, SRT, or JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from travidia.core.models import TranscriptionResult


def write_txt(result: TranscriptionResult, output_path: Path) -> None:
    """Write one line per segment, speaker-prefixed when known."""
    lines = []
    for segment in result.segments:
        if segment.speaker:
            lines.append(f"[{segment.speaker}] {segment.text}")
        else:
            lines.append(segment.text)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(result: TranscriptionResult, output_path: Path) -> None:
    """Write segments as standard SubRip (.srt)."""
    blocks = []
    for index, segment in enumerate(result.segments, start=1):
        start = _format_srt_timestamp(segment.start)
        end = _format_srt_timestamp(segment.end)
        blocks.append(f"{index}\n{start} --> {end}\n{segment.text}")

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_json(result: TranscriptionResult, output_path: Path) -> None:
    """Serialize the full result (segments + language) as JSON."""
    data = {
        "language": result.language,
        "segments": [asdict(segment) for segment in result.segments],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT's HH:MM:SS,mmm timestamp."""
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
