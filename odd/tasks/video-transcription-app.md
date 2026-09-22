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
- [x] T5 — CLI: `cli/main.py` (Typer app, `transcribe` command with `--output`, `--diarize`, `--model` flags). Route: delegated writer. Depends on T2-T4.
- [x] T6 — Web: `web/app.py` (FastAPI: upload endpoint, transcribe job, diarize toggle) + `web/static/index.html` (minimal upload UI). Route: delegated writer. Depends on T2-T4.
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
- 2026-09-22: T5 done (CLI). `src/travidia/cli/main.py`: Typer app, `travidia transcribe VIDEO_PATH --output OUTPUT [--diarize] [--model] [--device] [--compute-type]`. Extracts audio into a `tempfile.TemporaryDirectory()` (auto-cleaned), validates `HF_TOKEN` env var before calling `transcribe()` when `--diarize` is set (fails fast, exit 1, no core call), infers export format from `OUTPUT`'s suffix (`.txt`/`.srt`/`.json` → `write_txt`/`write_srt`/`write_json`; unknown suffix → exit 1), converts `AudioExtractionError` and any transcription exception into a clean stderr message + exit 1 (no raw traceback), prints `Transcription written to <path>` on success.
  - Gotcha: Typer collapses a `Typer()` app with exactly one `@app.command()` into an implicit top-level command (no subcommand name needed), which conflicts with the required `travidia transcribe ...` invocation shape. Fixed by adding an empty `@app.callback()` — that forces Typer to build a command Group, so `transcribe` stays a real subcommand. Verified via `typer.main.get_command` source (collapses only when `registered_callback`/`info.callback`/`registered_groups` are all falsy and exactly 1 command is registered).
  - TDD: RED confirmed (`ModuleNotFoundError: travidia.cli.main`) before implementation, then GREEN. Mocks patch `travidia.cli.main.extract_audio`/`travidia.cli.main.transcribe` (not `travidia.core.*`), via `typer.testing.CliRunner`.
  - Commit: `91d64d8`. `pytest tests/cli -v`: 5 passed, 0 failed. Full suite (`pytest -q`): 23 passed.
- 2026-09-22: T6 done (web app), delegated writer. `src/travidia/web/app.py`: FastAPI app with `GET /` (serves `web/static/index.html` via `FileResponse`) and `POST /api/transcribe` (multipart: `video`, `diarize`, `output_format`, `model_name`, `device`, `compute_type`). Mirrors `cli/main.py`'s behavior over HTTP: invalid `output_format` → `HTTPException(400)`; `diarize=true` with missing/empty `HF_TOKEN` env var → `HTTPException(400, "HF_TOKEN environment variable is required for diarization")`, checked before any core call; upload saved into a `tempfile.TemporaryDirectory()` → `extract_audio` → `transcribe` → matching `write_txt`/`write_srt`/`write_json`; `AudioExtractionError`/transcription exceptions → clean `HTTPException(500, ...)`, no raw traceback. `src/travidia/web/static/index.html`: single-page vanilla HTML/CSS/JS form (file input, diarize checkbox, format select, submit) that POSTs via `fetch`+`FormData` and triggers a browser download via `Blob`+temporary `<a>` click, with inline error display on non-OK responses.
  - Cleanup strategy: in-memory read, not `BackgroundTask`. Inside the `with tempfile.TemporaryDirectory()` block, the exported file's bytes are read (`output_path.read_bytes()`) before the block exits, so the temp dir (uploaded video, extracted WAV, exported file) is removed synchronously before the `Response` is built — avoids any lingering temp dir on client disconnect and ordering subtleties around background cleanup after send.
  - Mocking pattern mirrors `tests/cli/test_main.py`: patch `travidia.web.app.extract_audio`/`travidia.web.app.transcribe` (import site, not `travidia.core.*`); mocked `transcribe` returns a real `TranscriptionResult`/`Segment` so the real `write_*` export functions run and tests assert on actual response bytes.
  - Gotcha: none blocking — `fastapi==0.141.1`/`starlette==1.6.0`'s `TestClient` (httpx-backed) worked out of the box; `bool = Form(...)` parses `"true"`/`"false"` from form data correctly with no special handling. Only a harmless `StarletteDeprecationWarning` (suggesting `httpx2`) appears in test output, no functional impact.
  - TDD: RED confirmed (`ModuleNotFoundError: travidia.web.app`) before implementation, then GREEN on first pass after implementing.
  - Commit: `0cf8bc1`. `pytest tests/web -v`: 9 passed, 0 failed. Full suite (`pytest -q`): 32 passed, 0 failed (re-verified independently).
  - T7 (manual end-to-end verification with real HF token and GPU, both CLI and web, with/without diarization) remains open — requires the user.
