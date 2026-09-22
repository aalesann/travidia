# travidia

Local video transcription with optional speaker diarization, GPU-accelerated
via [WhisperX](https://github.com/m-bain/whisperX). Audio never leaves the
machine — no cloud APIs, no uploads to a third party.

Use it from the command line or from a local single-page web app; both are
thin wrappers over the same transcription engine.

## Features

- Extracts audio from any video ffmpeg can read.
- Transcribes with WhisperX (built on faster-whisper/CTranslate2) on GPU.
- Optional speaker diarization (pyannote.audio) — know who said what.
- Exports to `.txt`, `.srt`, or `.json`.
- Two interfaces, one engine: CLI (`travidia transcribe ...`) and a local web
  UI (`uvicorn travidia.web.app:app`).

## Requirements

- An NVIDIA GPU with CUDA support. Tested on an RTX 3060 Laptop (6GB VRAM),
  which is enough for `large-v3` with fp16.
- Python `>=3.10,<3.14`.
- `ffmpeg` available on `PATH`.
- A free [HuggingFace](https://huggingface.co/) account and access token —
  only needed for diarization (see below).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This pulls in PyTorch, WhisperX, and pyannote.audio — a multi-gigabyte
download; the first install can take several minutes.

### Diarization setup (optional)

Diarization uses a gated pyannote.audio model. One-time setup:

1. Accept the model license at
   https://huggingface.co/pyannote/speaker-diarization-3.1.
2. Create a read-only access token at
   https://huggingface.co/settings/tokens.
3. Export it in your shell — never commit it or pass it in chat/logs:

   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

After the model downloads once, diarization runs fully offline.

## Usage

### CLI

```bash
travidia transcribe video.mp4 --output out.srt
travidia transcribe video.mp4 --output out.srt --diarize
```

Output format is inferred from `--output`'s extension (`.txt`, `.srt`,
`.json`). Other flags:

| Flag             | Default     | Description                              |
|------------------|-------------|-------------------------------------------|
| `--diarize`      | off         | Enable speaker diarization (needs `HF_TOKEN`) |
| `--model`        | `large-v3`  | WhisperX/faster-whisper model to load    |
| `--device`       | `cuda`      | Torch device (`cuda` or `cpu`)           |
| `--compute-type` | `float16`   | Model precision (e.g. `float16`, `int8`) |

Running on CPU works too, just slower — no code change needed, only the
flags: `float16` isn't supported by CTranslate2 on CPU, so pair `--device
cpu` with `--compute-type int8`:

```bash
travidia transcribe video.mp4 --output out.srt --device cpu --compute-type int8
```

### Web (local)

```bash
uvicorn travidia.web.app:app --reload
```

Open http://localhost:8000, upload a video, pick a format, toggle
diarization, choose GPU or CPU, submit — the transcript downloads as a file.

## Project structure

```
src/travidia/
├── core/          # audio extraction, transcription, diarization, export
├── cli/           # Typer CLI, thin wrapper over core
└── web/           # FastAPI app + static frontend, thin wrapper over core
tests/             # pytest, mirrors src/travidia structure
```

## Testing

```bash
pytest
```

Unit tests mock WhisperX/pyannote and ffmpeg subprocess calls — they run
fast, offline, and without a GPU. No test downloads real models.
