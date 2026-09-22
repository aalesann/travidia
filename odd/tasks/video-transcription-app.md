# Feature: video-transcription-app

## Objective
Local app to transcribe video files, with optional speaker diarization, selectable via CLI or a local web UI. Everything runs on-device (RTX 3060 Laptop, 6GB VRAM); no audio/video leaves the machine.

## Why
User wants to transcribe videos locally, choosing per-run whether to enable speaker diarization, from either a CLI or a local web app — sharing one transcription engine.

## Constraints / verified facts
- GPU: RTX 3060 Laptop, 6GB VRAM, driver 615.71.09.
- Python 3.13.5 (system, only version available — no pyenv/uv installed).
- ffmpeg 7.1.5 present.
- whisperx 3.8.6 (PyPI) requires Python `>=3.10,<3.14` — 3.13.5 is compatible. Pulls faster-whisper, ctranslate2, pyannote-audio>=4.0.0, torch~=2.8.0.
- Diarization needs a HuggingFace token (gated pyannote model), one-time, local after download.

## Scope
- `src/travidia/core`: audio extraction (ffmpeg), transcription (WhisperX), diarization toggle, export (txt/srt/json).
- `src/travidia/cli`: Typer CLI, thin wrapper over core.
- `src/travidia/web`: FastAPI app + minimal HTML/JS frontend, thin wrapper over core.
- Tests for all of the above (pytest, mocking heavy model calls — no real GPU inference in unit tests).

## TDD
Strict TDD mode: enabled (source: user's global CLAUDE.md). Runner: pytest. RED (failing test) before implementation, then GREEN, then REFACTOR, per task.

## Tasks

- [x] T1 — Scaffolding: git repo, feature branch, directory layout, `pyproject.toml`, `.gitignore`, `README.md`, package `__init__.py` files. Route: direct inline (mechanical, no design decisions).
- [x] T2 — Core: `audio.py` (extract audio track from video via ffmpeg subprocess). Route: delegated writer (implementation + test, non-trivial).
- [x] T3 — Core: `transcribe.py` (WhisperX wrapper: load model, transcribe, optional diarization via pyannote, on GPU). Route: delegated writer.
- [x] T4 — Core: `export.py` (write results as txt/srt/json). Route: delegated writer.
- [ ] T5 — CLI: `cli/main.py` (Typer app, `transcribe` command with `--output`, `--diarize`, `--model` flags). Route: delegated writer. Depends on T2-T4.
- [ ] T6 — Web: `web/app.py` (FastAPI: upload endpoint, transcribe job, diarize toggle) + `web/static/index.html` (minimal upload UI). Route: delegated writer. Depends on T2-T4.
- [ ] T7 — Manual verification: real end-to-end run with actual video, both CLI and web, with and without diarization (requires user's HF token — cannot be fully automated here).

## Delivery
Delivery strategy: not yet chosen (forecast is well under ~400 changed-line budget for now; will revisit if it grows).

## Progress log
- 2026-09-22: T1 done, direct inline. Verified whisperx/Python 3.13 compatibility via PyPI metadata before scaffolding.
- 2026-09-22: T2+T3+T4 done (core transcription engine). Created `.venv`, `pip install -e ".[dev]"` (torch 2.8.0, whisperx 3.8.6, pyannote-audio 4.0.7 — several minutes, no install errors on Python 3.13.5). Strict TDD followed per module: RED confirmed (ModuleNotFoundError) before each implementation, then GREEN.
  - `src/travidia/core/models.py`: `Segment`/`TranscriptionResult` dataclasses.
  - `src/travidia/core/audio.py`: `extract_audio()` via `subprocess.run(["ffmpeg", ...])`, mono 16kHz WAV, `AudioExtractionError` on missing input or non-zero ffmpeg exit (stderr included). Tests mock `subprocess.run` (no real ffmpeg decode needed for the logic under test).
  - `src/travidia/core/transcribe.py`: `transcribe()` wraps `whisperx.load_model` → `model.transcribe` → `whisperx.load_align_model`/`whisperx.align` → optional `whisperx.diarize.DiarizationPipeline`/`whisperx.assign_word_speakers`. `whisperx` imported at module level; tests use `unittest.mock.patch("travidia.core.transcribe.whisperx")`, no real model download/GPU use.
  - `src/travidia/core/export.py`: `write_txt`/`write_srt`/`write_json`.
  - Gotcha: installed whisperx 3.8.6's `DiarizationPipeline.__init__` takes `token=`, not `use_auth_token=` as the task description assumed — verified via `inspect.signature` against the installed package and used `token=` in both implementation and test. Audio is passed to WhisperX calls as a path string (transcribe/align/diarize all accept `str`), so `whisperx.load_audio` was not needed.
  - Commits: `d2ff44b` (audio+models), `2c51dd8` (transcribe), `9070779` (export). `pytest tests/core -v`: 18 passed, 0 failed.
