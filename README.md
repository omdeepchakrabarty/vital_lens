# Vital Lens rPPG

A lightweight web app for remote photoplethysmography (rPPG) with a FastAPI backend and vanilla JS frontend.

## Architecture

- **Frontend** (`frontend/`): webcam capture, recording, upload, and display of only **BPM / HRV / SBP / DBP**.
- **Backend** (`backend/`): upload API, decode pipeline, deep-model inference pipeline.
- **Traditional modules** (`POS`, `CHROM`, `Green`) are included and executed for diagnostics, while returned vitals are produced exclusively by the deep model output path.

## Quick Start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://localhost:8000`

## Docker

```bash
docker compose up --build
```

## API

- `GET /api/health`
- `POST /api/process` with multipart video file field: `file`

## Model file handling

Expected model path:

`backend/app/models/BP4D_BigSmall_Multitask_Fold2.pth`

Configure strict behavior:

- `MODEL_STRICT_LOADING=true` => startup fails when file is missing.
- default false => app starts and uses initialized network parameters.

## Notes

This application is intended for software demonstration and pipeline validation only, not medical diagnosis.
