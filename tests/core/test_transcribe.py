"""Tests for the WhisperX transcription wrapper.

Approach: `whisperx` is imported at module level in `travidia.core.transcribe`
and every call goes through that module attribute, so tests replace it with
`unittest.mock.patch("travidia.core.transcribe.whisperx")`. This avoids
downloading any real WhisperX/pyannote models or touching a GPU, keeping the
suite fast, offline, and deterministic.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from travidia.core.models import TranscriptionResult
from travidia.core.transcribe import transcribe


def _configure_whisperx_mock(mock_whisperx: MagicMock):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
        "language": "en",
    }
    mock_whisperx.load_model.return_value = mock_model

    mock_align_model = MagicMock()
    mock_metadata = MagicMock()
    mock_whisperx.load_align_model.return_value = (mock_align_model, mock_metadata)
    mock_whisperx.align.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
    }
    return mock_whisperx


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_returns_transcription_result(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    result = transcribe(audio_path)

    assert isinstance(result, TranscriptionResult)
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "hello"
    assert result.segments[0].speaker is None


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_loads_model_with_requested_params(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    transcribe(audio_path, model_name="small", device="cpu", compute_type="int8")

    mock_whisperx.load_model.assert_called_once()
    _, kwargs = mock_whisperx.load_model.call_args
    call_args = mock_whisperx.load_model.call_args.args
    assert "small" in call_args or kwargs.get("model_name") == "small" or "small" in call_args
    assert "cpu" in call_args or kwargs.get("device") == "cpu"


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_runs_alignment(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    transcribe(audio_path)

    mock_whisperx.load_align_model.assert_called_once()
    mock_whisperx.align.assert_called_once()


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_diarize_without_token_raises_value_error(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    with pytest.raises(ValueError, match="hf_token"):
        transcribe(audio_path, diarize=True, hf_token=None)

    with pytest.raises(ValueError, match="hf_token"):
        transcribe(audio_path, diarize=True, hf_token="")


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_diarize_assigns_speakers(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    mock_diarize_pipeline_instance = MagicMock()
    mock_diarize_pipeline_instance.return_value = "diarize_segments"
    mock_whisperx.diarize.DiarizationPipeline.return_value = mock_diarize_pipeline_instance

    mock_whisperx.assign_word_speakers.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "SPEAKER_01"},
        ]
    }

    result = transcribe(audio_path, diarize=True, hf_token="hf_fake_token")

    mock_whisperx.diarize.DiarizationPipeline.assert_called_once()
    _, kwargs = mock_whisperx.diarize.DiarizationPipeline.call_args
    assert kwargs.get("token") == "hf_fake_token"
    mock_whisperx.assign_word_speakers.assert_called_once()

    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"


@patch("travidia.core.transcribe.whisperx")
def test_transcribe_does_not_diarize_by_default(mock_whisperx: MagicMock, tmp_path: Path):
    _configure_whisperx_mock(mock_whisperx)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    transcribe(audio_path)

    mock_whisperx.diarize.DiarizationPipeline.assert_not_called()
    mock_whisperx.assign_word_speakers.assert_not_called()
