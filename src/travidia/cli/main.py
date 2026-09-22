"""Typer CLI for travidia: local video transcription."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import typer

from travidia.core.audio import AudioExtractionError, extract_audio
from travidia.core.export import write_json, write_srt, write_txt
from travidia.core.models import TranscriptionResult
from travidia.core.transcribe import transcribe

app = typer.Typer()

_EXPORTERS = {
    ".txt": write_txt,
    ".srt": write_srt,
    ".json": write_json,
}


@app.callback()
def callback() -> None:
    """travidia: local video transcription with optional speaker diarization."""


@app.command("transcribe")
def transcribe_command(
    video_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the source video file.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path. Format is inferred from the suffix (.txt, .srt, .json).",
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Enable speaker diarization. Requires the HF_TOKEN environment variable.",
    ),
    model: str = typer.Option(
        "large-v3",
        "--model",
        help="WhisperX/faster-whisper model name.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Torch device to run on (e.g. cuda or cpu).",
    ),
    compute_type: str = typer.Option(
        "float16",
        "--compute-type",
        help="Compute precision for the Whisper model (e.g. float16, int8).",
    ),
) -> None:
    """Transcribe VIDEO_PATH and write the result to OUTPUT."""
    exporter = _EXPORTERS.get(output.suffix.lower())
    if exporter is None:
        typer.echo(
            f"Error: unsupported output format '{output.suffix}'. "
            f"Supported formats: {', '.join(sorted(_EXPORTERS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    hf_token: str | None = None
    if diarize:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            typer.echo(
                "Error: --diarize requires the HF_TOKEN environment variable to be set.",
                err=True,
            )
            raise typer.Exit(code=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            audio_path = extract_audio(video_path, output_dir=Path(tmp_dir))
        except AudioExtractionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        try:
            result: TranscriptionResult = transcribe(
                audio_path,
                model_name=model,
                diarize=diarize,
                hf_token=hf_token,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:  # noqa: BLE001 - convert any transcription failure to a clean CLI error
            typer.echo(f"Error: transcription failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        exporter(result, output)

    typer.echo(f"Transcription written to {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
