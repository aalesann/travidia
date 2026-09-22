# travidia

Local video transcription with optional speaker diarization. GPU-accelerated
via WhisperX, no audio ever leaves the machine.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Diarization uses pyannote.audio, whose models are gated on HuggingFace. Accept
the model license once at https://huggingface.co/pyannote/speaker-diarization-3.1
and export a token before running with `--diarize`:

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

CLI:

```bash
travidia transcribe video.mp4 --output out.srt
travidia transcribe video.mp4 --output out.srt --diarize
```

Web (local):

```bash
uvicorn travidia.web.app:app --reload
```

Then open http://localhost:8000.
