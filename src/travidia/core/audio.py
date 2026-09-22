"""ffmpeg-based audio extraction from video files.

Extracts a mono 16kHz WAV track (the input format WhisperX/faster-whisper
expects) via an `ffmpeg` subprocess. The extracted file defaults to living
next to the source video; pass `output_dir` to redirect it (e.g. to a temp
directory) instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from a video file."""


def extract_audio(video_path: Path, output_dir: Path | None = None) -> Path:
    """Extract mono 16kHz WAV audio from `video_path` via ffmpeg.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to write the extracted WAV into. Defaults to
            the video's own directory. Created if it does not exist.

    Returns:
        Path to the extracted `<video-stem>.wav` file.

    Raises:
        AudioExtractionError: if `video_path` does not exist, or ffmpeg
            exits with a non-zero return code (ffmpeg's stderr is included
            in the error message).
    """
    if not video_path.exists():
        raise AudioExtractionError(f"Input video file does not exist: {video_path}")

    target_dir = output_dir if output_dir is not None else video_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{video_path.stem}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed to extract audio from {video_path} "
            f"(exit code {result.returncode}): {result.stderr}"
        )

    return output_path
