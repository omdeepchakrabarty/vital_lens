from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.core.config import get_settings


class BigSmallMultitaskNet(nn.Module):
    """Compact inference network compatible with multitask regression output."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class ModelInferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.model_device)
        self.model = BigSmallMultitaskNet().to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        model_path = self.settings.model_path
        if not model_path.exists():
            if self.settings.model_strict_loading:
                raise FileNotFoundError(f"Required model file not found: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=False)

    @torch.inference_mode()
    def predict_metrics_from_trace(self, rgb_trace: np.ndarray) -> dict[str, float]:
        x = torch.tensor(rgb_trace, dtype=torch.float32, device=self.device).T.unsqueeze(0)
        pred = self.model(x).squeeze(0).detach().cpu().numpy()

        bpm = 75.0 + pred[0] * 10.0
        hrv = 45.0 + pred[1] * 12.0
        sbp = 118.0 + pred[2] * 9.0
        dbp = 76.0 + pred[3] * 7.0

        return {"bpm": float(bpm), "hrv": float(hrv), "sbp": float(sbp), "dbp": float(dbp)}
