from dataclasses import dataclass


@dataclass
class VitalMetrics:
    bpm: float
    hrv: float
    sbp: float
    dbp: float


def sanitize_metrics(raw: dict[str, float]) -> VitalMetrics:
    bpm = float(max(40.0, min(190.0, raw["bpm"])))
    hrv = float(max(10.0, min(220.0, raw["hrv"])))
    sbp = float(max(85.0, min(180.0, raw["sbp"])))
    dbp = float(max(50.0, min(120.0, raw["dbp"])))
    return VitalMetrics(bpm=bpm, hrv=hrv, sbp=sbp, dbp=dbp)
