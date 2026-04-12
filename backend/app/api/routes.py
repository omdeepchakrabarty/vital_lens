from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.metrics_postprocess import postprocess_metrics
from app.services.model_inference import ModelInferenceService
from app.services.video_ingest import decode_video_to_rgb_trace
from app.traditional.chrom import extract_chrom_signal
from app.traditional.green import extract_green_signal
from app.traditional.pos import extract_pos_signal

router = APIRouter()
model_service = ModelInferenceService()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/process")
async def process_video(file: UploadFile = File(...)) -> dict:
    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    payload = await file.read()

    try:
        rgb_trace = decode_video_to_rgb_trace(payload)

        # Traditional pipelines are executed for observability/troubleshooting.
        traditional_debug = {
            "green_energy": float((extract_green_signal(rgb_trace) ** 2).mean()),
            "chrom_energy": float((extract_chrom_signal(rgb_trace) ** 2).mean()),
            "pos_energy": float((extract_pos_signal(rgb_trace) ** 2).mean()),
        }

        # Returned vitals are derived exclusively from deep model inference.
        raw_outputs = model_service.predict_metrics_from_video(payload)
        metrics = postprocess_metrics(raw_outputs)

        return {
            "metrics": metrics,
            "diagnostics": {"traditional_pipeline": traditional_debug},
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {exc}") from exc
