import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.metrics_postprocess import postprocess_metrics
from app.services.model_inference import ModelInferenceService
from app.services.video_ingest import decode_video_to_frames_and_trace
from app.traditional.chrom import CHROMProcessor
from app.traditional.green import GreenChannelProcessor
from app.traditional.pos import POSProcessor

router = APIRouter()
model_service = ModelInferenceService()
logger = logging.getLogger(__name__)
pos_processor = POSProcessor()
chrom_processor = CHROMProcessor()
green_processor = GreenChannelProcessor()


def _dominant_frequency_hz(signal: "np.ndarray", fps: float = 30.0) -> float:
    if signal.size < 4:
        return 0.0
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fps)
    if spectrum.size <= 1:
        return 0.0
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[peak_idx])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/process")
async def process_video(file: UploadFile = File(...)) -> dict:
    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    payload = await file.read()

    try:
        # Decode once, then run both auxiliary traditional streams and DL inference.
        frames_rgb, rgb_trace = decode_video_to_frames_and_trace(payload)

        # These streams are intentionally kept visible as ensemble-like validation
        # paths for diagnostics only; they do not affect returned API vitals.
        pos_signal = pos_processor.process(frames_rgb)
        chrom_signal = chrom_processor.process(frames_rgb)
        green_signal = green_processor.process(frames_rgb)

        traditional_debug = {
            "signal_lengths": {
                "pos": int(pos_signal.shape[0]),
                "chrom": int(chrom_signal.shape[0]),
                "green": int(green_signal.shape[0]),
            },
            "spectral_peaks_hz": {
                "pos": _dominant_frequency_hz(pos_signal),
                "chrom": _dominant_frequency_hz(chrom_signal),
                "green": _dominant_frequency_hz(green_signal),
            },
        }
        logger.debug("Traditional auxiliary streams computed: %s", traditional_debug)

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
