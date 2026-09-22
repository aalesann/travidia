"""FastAPI app for local, single-user video transcription over HTTP.

Mirrors the behavior of `travidia.cli.main`: same temp-dir handling via
`tempfile.TemporaryDirectory()`, the same HF_TOKEN-from-env fail-fast check
before calling `transcribe()` when diarization is requested, the same
format-from-selection export dispatch, and the same conversion of
`AudioExtractionError`/transcription exceptions into clean user-facing
errors (here, `HTTPException` instead of `typer.Exit`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from travidia.core.audio import AudioExtractionError, extract_audio
from travidia.core.export import write_json, write_srt, write_txt
from travidia.core.models import TranscriptionResult
from travidia.core.transcribe import transcribe

app = FastAPI(title="travidia")

_STATIC_DIR = Path(__file__).parent / "static"

_EXPORTERS = {
    "txt": write_txt,
    "srt": write_srt,
    "json": write_json,
}

_MEDIA_TYPES = {
    "txt": "text/plain",
    "srt": "text/plain",
    "json": "application/json",
}


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page web UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/api/transcribe")
async def transcribe_endpoint(
    video: UploadFile,
    diarize: bool = Form(False),
    output_format: str = Form("srt"),
    model_name: str = Form("large-v3"),
    device: str = Form("cuda"),
    compute_type: str = Form("float16"),
) -> Response:
    """Transcribe an uploaded video and return the exported result as a download."""
    exporter = _EXPORTERS.get(output_format)
    if exporter is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported output format '{output_format}'. "
                f"Supported formats: {', '.join(sorted(_EXPORTERS))}"
            ),
        )

    hf_token: str | None = None
    if diarize:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=400,
                detail="HF_TOKEN environment variable is required for diarization",
            )

    # The temp dir is used for the uploaded video, the extracted audio, and
    # the exported output file. The exported file's bytes are read into
    # memory before the `with` block exits, so the directory (and every file
    # in it) is cleaned up synchronously before the response is returned --
    # no BackgroundTask is needed and no temp file outlives this request.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        video_path = tmp_path / (video.filename or "upload.mp4")
        video_bytes = await video.read()
        video_path.write_bytes(video_bytes)

        try:
            audio_path = extract_audio(video_path, output_dir=tmp_path)
        except AudioExtractionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            result: TranscriptionResult = transcribe(
                audio_path,
                model_name=model_name,
                diarize=diarize,
                hf_token=hf_token,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:  # noqa: BLE001 - convert any transcription failure to a clean HTTP error
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        output_filename = f"transcript.{output_format}"
        output_path = tmp_path / output_filename
        exporter(result, output_path)
        content = output_path.read_bytes()

    return Response(
        content=content,
        media_type=_MEDIA_TYPES[output_format],
        headers={"Content-Disposition": f"attachment; filename={output_filename}"},
    )
