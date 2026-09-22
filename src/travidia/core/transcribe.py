"""WhisperX-based transcription with optional speaker diarization.

`whisperx` is imported at module level and every call goes through this
module's `whisperx` attribute, so tests can replace it wholesale with
`unittest.mock.patch("travidia.core.transcribe.whisperx")` instead of
downloading real models or touching a GPU.

Audio is passed to WhisperX/pyannote as a path string rather than a
pre-loaded array (WhisperX's `transcribe`/`align`/diarization calls all
accept a `str` path directly), which keeps this wrapper simple.
"""

from __future__ import annotations

from pathlib import Path

import whisperx
import whisperx.diarize

from travidia.core.models import Segment, TranscriptionResult


def transcribe(
    audio_path: Path,
    *,
    model_name: str = "large-v3",
    diarize: bool = False,
    hf_token: str | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
) -> TranscriptionResult:
    """Transcribe `audio_path` with WhisperX, optionally diarizing speakers.

    Args:
        audio_path: Path to the (mono 16kHz WAV) audio file to transcribe.
        model_name: WhisperX/faster-whisper model to load (e.g. "large-v3").
        diarize: If True, run pyannote speaker diarization and assign a
            `speaker` label to each segment. Requires `hf_token`.
        hf_token: HuggingFace access token for the gated pyannote model.
            Required when `diarize=True`.
        device: Torch device to run on (e.g. "cuda" or "cpu").
        compute_type: Compute precision for the Whisper model (e.g.
            "float16", "int8").

    Returns:
        A `TranscriptionResult` with aligned (and optionally
        speaker-labeled) segments.

    Raises:
        ValueError: if `diarize=True` and `hf_token` is missing or empty.
    """
    if diarize and not hf_token:
        raise ValueError("hf_token is required when diarize=True")

    audio = str(audio_path)

    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    transcription = model.transcribe(audio)
    language = transcription["language"]

    align_model, align_metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        transcription["segments"],
        align_model,
        align_metadata,
        audio,
        device,
    )

    segments_data = aligned["segments"]

    if diarize:
        diarize_pipeline = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize_pipeline(audio)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)
        segments_data = result_with_speakers["segments"]

    segments = [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip() if isinstance(seg.get("text"), str) else seg["text"],
            speaker=seg.get("speaker"),
        )
        for seg in segments_data
    ]

    return TranscriptionResult(segments=segments, language=language)
