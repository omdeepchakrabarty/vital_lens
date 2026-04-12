# Vital Lens rPPG

Web-based remote photoplethysmography (rPPG) demo with:

- **FastAPI backend** for video upload + inference.
- **Vanilla JS frontend** for camera capture and metric display.
- **Deep-learning-only output path** for returned vitals.
- **Traditional POS/CHROM/Green modules** kept in pipeline diagnostics for observability and architectural parity.

---

## Repository Layout (exact)

```text
vital_lens/
├── .devcontainer/
│   └── devcontainer.json
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes.py
│       ├── core/
│       │   └── config.py
│       ├── services/
│       │   ├── metrics_postprocess.py
│       │   ├── model_inference.py
│       │   └── video_ingest.py
│       ├── traditional/
│       │   ├── chrom.py
│       │   ├── green.py
│       │   └── pos.py
│       └── main.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── models/
│   └── BP4D_BigSmall_Multitask_Fold2.pth
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model Placement

Place your trained checkpoint file at:

```text
models/BP4D_BigSmall_Multitask_Fold2.pth
```

The backend resolves this from project root (`PROJECT_ROOT` env override, then sentinel discovery, then cwd contract) using `model_relative_path` in `backend/app/core/config.py`.

### Strict startup behavior (optional)

- Default: `MODEL_STRICT_LOADING=false` (service starts even if weights are missing).
- Strict mode: set `MODEL_STRICT_LOADING=true` to fail fast when checkpoint is absent.

Example:

```bash
export MODEL_STRICT_LOADING=true
```

---

## Local Run (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- App UI: `http://localhost:8000`
- Health: `http://localhost:8000/api/health`

---

## Local Run (Docker)

```bash
docker build -t vital-lens .
docker run --rm -p 8000:8000 vital-lens
```

or:

```bash
docker compose up --build
```

---

## GitHub Codespaces Run Instructions

This repo includes `.devcontainer/devcontainer.json` and Dockerfile-based setup.

1. Open the repo in **GitHub Codespaces**.
2. Wait for container build + dependency installation to finish.
3. In Codespaces terminal, start backend:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Open forwarded port **8000** from the **Ports** panel.
5. Use the browser preview to access the frontend UI.

---

## API Reference

### `GET /api/health`

**Response**

```json
{
  "status": "ok"
}
```

### `POST /api/process`

- Content-Type: `multipart/form-data`
- Form field name: `file`
- Accepted input: browser-recorded video (`video/webm`, etc.)

#### cURL example

```bash
curl -X POST "http://localhost:8000/api/process" \
  -H "accept: application/json" \
  -F "file=@capture.webm;type=video/webm"
```

#### Example successful response

```json
{
  "metrics": {
    "bpm": 76.8,
    "hrv": 43.2,
    "sbp": 121.4,
    "dbp": 77.0
  },
  "diagnostics": {
    "traditional_pipeline": {
      "green_energy": 0.0023,
      "chrom_energy": 0.0019,
      "pos_energy": 0.0015
    }
  }
}
```

#### Example error response

```json
{
  "detail": "Uploaded file must be a video."
}
```

---

## Frontend Usage Flow

1. Open the app in browser (`/`).
2. Allow camera permission.
3. Click **Start Recording**.
4. Frontend records ~6 seconds (`MediaRecorder`), then stops automatically.
5. Browser uploads the clip to `POST /api/process` as multipart form data.
6. Backend decodes frames, runs traditional diagnostics (POS/CHROM/Green), runs deep model inference, and returns metrics.
7. Frontend displays:
   - BPM
   - HRV
   - SBP
   - DBP

---

## Notes

- Traditional algorithms are intentionally retained in backend diagnostics to support visibility and troubleshooting of signal characteristics.
- **Returned vitals are derived from the deep learning inference output path only.**
- For research/demo use only; not intended for clinical diagnosis.
