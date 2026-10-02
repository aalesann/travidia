"""Tests for the FastAPI web app (`travidia.web.app`).

Approach: `extract_audio` and `transcribe` are mocked where they are imported
into `travidia.web.app` (not in `travidia.core.*`), so tests never touch
ffmpeg, WhisperX, or a GPU. The mocked `transcribe` returns a real
`TranscriptionResult` built from the actual dataclasses, so the real
`write_txt`/`write_srt`/`write_json` functions run for real and produce
real bytes to assert against.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from travidia.core.audio import AudioExtractionError
from travidia.core.export import write_json, write_srt, write_txt
from travidia.core.models import Segment, TranscriptionResult
from travidia.web.app import app

client = TestClient(app)


def _fake_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            Segment(start=0.0, end=1.5, text="hello world", speaker="SPEAKER_00"),
            Segment(start=1.5, end=3.0, text="second segment"),
        ],
        language="en",
    )


def _post_transcribe(**form):
    form.setdefault("output_format", "srt")
    files = {"video": ("video.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")}
    return client.post("/api/transcribe", data=form, files=files)


def test_get_root_serves_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_transcribe_success_srt(tmp_path: Path):
    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.return_value = _fake_result()

        response = _post_transcribe(output_format="srt")

    assert response.status_code == 200, response.text
    assert "attachment" in response.headers["content-disposition"]
    assert "filename=" in response.headers["content-disposition"]
    assert response.headers["content-type"].startswith(("text/plain", "application/x-subrip"))

    expected_path = tmp_path / "expected.srt"
    write_srt(_fake_result(), expected_path)
    assert response.content == expected_path.read_bytes()

    mock_extract_audio.assert_called_once()
    mock_transcribe.assert_called_once()


def test_transcribe_success_txt(tmp_path: Path):
    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.return_value = _fake_result()

        response = _post_transcribe(output_format="txt")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")

    expected_path = tmp_path / "expected.txt"
    write_txt(_fake_result(), expected_path)
    assert response.content == expected_path.read_bytes()


def test_transcribe_success_json(tmp_path: Path):
    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.return_value = _fake_result()

        response = _post_transcribe(output_format="json")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")

    expected_path = tmp_path / "expected.json"
    write_json(_fake_result(), expected_path)
    assert json.loads(response.content) == json.loads(expected_path.read_text(encoding="utf-8"))


def test_transcribe_diarize_without_hf_token_fails(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        response = _post_transcribe(diarize="true")

    assert response.status_code == 400
    assert "HF_TOKEN" in response.json()["detail"]
    mock_extract_audio.assert_not_called()
    mock_transcribe.assert_not_called()


def test_transcribe_diarize_with_hf_token_succeeds(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.return_value = _fake_result()

        response = _post_transcribe(diarize="true")

    assert response.status_code == 200, response.text
    _, kwargs = mock_transcribe.call_args
    assert kwargs["diarize"] is True
    assert kwargs["hf_token"] == "fake-token"


def test_transcribe_invalid_output_format_fails():
    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        response = _post_transcribe(output_format="xyz")

    assert response.status_code == 400
    mock_extract_audio.assert_not_called()
    mock_transcribe.assert_not_called()


def test_transcribe_extract_audio_error_returns_clean_500():
    with patch("travidia.web.app.extract_audio") as mock_extract_audio:
        mock_extract_audio.side_effect = AudioExtractionError("ffmpeg exploded")

        response = _post_transcribe()

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "ffmpeg exploded" in detail
    assert "Traceback" not in detail


def test_transcribe_does_not_block_other_requests():
    """Regression test: the event loop must stay responsive while a
    transcription request is in flight, proving the blocking calls inside
    `transcribe_endpoint` run off the event loop (via `run_in_threadpool`)
    rather than directly on it.
    """

    def slow_transcribe(*args, **kwargs):
        time.sleep(0.3)  # simulates blocking WhisperX inference
        return _fake_result()

    async def run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            with (
                patch("travidia.web.app.extract_audio") as mock_extract_audio,
                patch("travidia.web.app.transcribe", side_effect=slow_transcribe),
            ):
                mock_extract_audio.return_value = Path("/tmp/fake.wav")
                files = {"video": ("video.mp4", io.BytesIO(b"fake"), "video/mp4")}
                transcribe_task = asyncio.create_task(
                    ac.post("/api/transcribe", data={"output_format": "srt"}, files=files)
                )
                # No sleep here on purpose: inserting an `await asyncio.sleep(...)`
                # between creating the task and issuing this GET would itself get
                # starved for the same reason this test exists -- if the event
                # loop is blocked by the synchronous transcribe call, that sleep
                # would not resume until the blocking call finishes, which would
                # silently swallow the very delay this test needs to observe.
                start = time.monotonic()
                root_response = await ac.get("/")
                elapsed = time.monotonic() - start
                transcribe_response = await transcribe_task

        assert root_response.status_code == 200
        assert elapsed < 0.2  # GET / returned well before the 0.3s transcribe finished
        assert transcribe_response.status_code == 200

    asyncio.run(run())


def test_transcribe_transcribe_error_returns_clean_500(tmp_path: Path):
    with (
        patch("travidia.web.app.extract_audio") as mock_extract_audio,
        patch("travidia.web.app.transcribe") as mock_transcribe,
    ):
        mock_extract_audio.return_value = tmp_path / "video.wav"
        mock_transcribe.side_effect = RuntimeError("model exploded")

        response = _post_transcribe()

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "model exploded" in detail
    assert "Traceback" not in detail
