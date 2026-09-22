"""Tests for the `travidia transcribe` CLI command.

Approach: `extract_audio` and `transcribe` are mocked where they are imported
into `travidia.cli.main` (not in `travidia.core.*`), so tests never touch
ffmpeg, WhisperX, or a GPU. `typer.testing.CliRunner` drives the app.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from travidia.cli.main import app
from travidia.core.audio import AudioExtractionError
from travidia.core.models import Segment, TranscriptionResult

runner = CliRunner()


def _fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[Segment(start=0.0, end=1.0, text="hello world")],
        language="en",
    )


def test_transcribe_success_writes_txt(tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_path = tmp_path / "output.txt"

    with (
        patch("travidia.cli.main.extract_audio") as mock_extract_audio,
        patch("travidia.cli.main.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.return_value = _fake_result()

        result = runner.invoke(
            app,
            ["transcribe", str(video_path), "--output", str(output_path)],
        )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "hello world" in output_path.read_text(encoding="utf-8")
    assert str(output_path) in result.output
    mock_extract_audio.assert_called_once()
    mock_transcribe.assert_called_once()


def test_transcribe_diarize_without_hf_token_fails(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_path = tmp_path / "output.txt"

    with (
        patch("travidia.cli.main.extract_audio") as mock_extract_audio,
        patch("travidia.cli.main.transcribe") as mock_transcribe,
    ):
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(video_path),
                "--output",
                str(output_path),
                "--diarize",
            ],
        )

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output
    mock_transcribe.assert_not_called()


def test_transcribe_unknown_output_suffix_fails(tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_path = tmp_path / "output.xyz"

    with (
        patch("travidia.cli.main.extract_audio") as mock_extract_audio,
        patch("travidia.cli.main.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"

        result = runner.invoke(
            app,
            ["transcribe", str(video_path), "--output", str(output_path)],
        )

    assert result.exit_code == 1
    assert "output" in result.output.lower()
    mock_transcribe.assert_not_called()


def test_transcribe_nonexistent_video_fails(tmp_path: Path):
    video_path = tmp_path / "does-not-exist.mp4"
    output_path = tmp_path / "output.txt"

    result = runner.invoke(
        app,
        ["transcribe", str(video_path), "--output", str(output_path)],
    )

    assert result.exit_code != 0


def test_transcribe_audio_extraction_error_fails_cleanly(tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video bytes")
    output_path = tmp_path / "output.txt"

    with patch("travidia.cli.main.extract_audio") as mock_extract_audio:
        mock_extract_audio.side_effect = AudioExtractionError("ffmpeg exploded")

        result = runner.invoke(
            app,
            ["transcribe", str(video_path), "--output", str(output_path)],
        )

    assert result.exit_code == 1
    assert "ffmpeg exploded" in result.output
