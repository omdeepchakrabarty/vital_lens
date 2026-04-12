from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VitalMetrics:
    hrv: float
    systolic_bp: float
    diastolic_bp: float
    bpm: float


def _safe_float(value: float, fallback: float) -> float:
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return fallback
    return numeric


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def postprocess_metrics(raw_inference: dict[str, list[float]]) -> dict[str, float]:
    raw_outputs = raw_inference.get("raw_outputs", [])
    if len(raw_outputs) < 4:
        raise ValueError("Model output does not contain required multitask predictions.")

    bpm = _safe_float(75.0 + (raw_outputs[0] * 10.0), 75.0)
    hrv = _safe_float(45.0 + (raw_outputs[1] * 12.0), 45.0)
    systolic_bp = _safe_float(118.0 + (raw_outputs[2] * 9.0), 118.0)
    diastolic_bp = _safe_float(76.0 + (raw_outputs[3] * 7.0), 76.0)

    metrics = VitalMetrics(
        hrv=_clamp(hrv, 10.0, 220.0),
        systolic_bp=_clamp(systolic_bp, 85.0, 200.0),
        diastolic_bp=_clamp(diastolic_bp, 50.0, 130.0),
        bpm=_clamp(bpm, 40.0, 190.0),
    )

    return {
        "hrv": metrics.hrv,
        "systolic_bp": metrics.systolic_bp,
        "diastolic_bp": metrics.diastolic_bp,
        "bpm": metrics.bpm,
    }
