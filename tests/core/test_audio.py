"""Tests for ffmpeg-based audio extraction.

Approach: mock `subprocess.run` for the ffmpeg invocation instead of
generating and decoding a real synthetic video. This keeps the unit tests
fast, offline, and deterministic, and avoids coupling the test suite to the
ffmpeg binary being present/behaving identically across machines. The
subprocess call surface (command, cwd-independent paths, returncode/stderr
handling) is simple enough that mocking it gives full coverage of our own
logic without needing a real decode.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from travidia.core.audio import AudioExtractionError, extract_audio


def test_extract_audio_raises_if_input_file_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist.mp4"

    with pytest.raises(AudioExtractionError, match="does not exist"):
        extract_audio(missing)


@patch("travidia.core.audio.subprocess.run")
def test_extract_audio_invokes_ffmpeg_with_expected_args(mock_run: MagicMock, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_dir = tmp_path / "out"

    mock_run.return_value = MagicMock(returncode=0, stderr="")

    result = extract_audio(video_path, output_dir=output_dir)

    assert result == output_dir / "input.wav"
    assert mock_run.call_count == 1
    args = mock_run.call_args.args[0]
    assert args[0] == "ffmpeg"
    assert str(video_path) in args
    assert str(result) in args
    # mono, 16kHz WAV
    assert "-ac" in args and args[args.index("-ac") + 1] == "1"
    assert "-ar" in args and args[args.index("-ar") + 1] == "16000"


@patch("travidia.core.audio.subprocess.run")
def test_extract_audio_defaults_output_dir_to_video_dir(mock_run: MagicMock, tmp_path: Path):
    video_path = tmp_path / "clip.mkv"
    video_path.write_bytes(b"fake video bytes")

    mock_run.return_value = MagicMock(returncode=0, stderr="")

    result = extract_audio(video_path)

    assert result == tmp_path / "clip.wav"


@patch("travidia.core.audio.subprocess.run")
def test_extract_audio_raises_on_ffmpeg_failure_with_stderr(mock_run: MagicMock, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake video bytes")

    mock_run.return_value = MagicMock(returncode=1, stderr="Invalid data found when processing input")

    with pytest.raises(AudioExtractionError, match="Invalid data found when processing input"):
        extract_audio(video_path, output_dir=tmp_path / "out")


@patch("travidia.core.audio.subprocess.run")
def test_extract_audio_creates_output_dir_if_missing(mock_run: MagicMock, tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_dir = tmp_path / "nested" / "out"

    mock_run.return_value = MagicMock(returncode=0, stderr="")

    extract_audio(video_path, output_dir=output_dir)

    assert output_dir.is_dir()
