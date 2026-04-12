from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class BigSmallMultitaskNet(nn.Module):
    """Compact BP4D-style multitask model interface for inference."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class ModelInferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.model_device)
        self.model = BigSmallMultitaskNet().to(self.device)
        self.input_size = (72, 72)
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._load_weights()
        self.model.eval()
        logger.info("Model initialized on device=%s", self.device)

    def _resolve_model_path(self) -> Path:
        env_override = (
            os.getenv("MODEL_PATH")
            or os.getenv("MODEL_CHECKPOINT_PATH")
            or os.getenv("BP4D_MODEL_PATH")
        )
        if env_override:
            return Path(env_override).expanduser().resolve()
        return self.settings.model_path

    def _load_weights(self) -> None:
        model_path = self._resolve_model_path()
        logger.info("Resolved model checkpoint path: %s", model_path)
        if not model_path.exists():
            if self.settings.model_strict_loading:
                raise FileNotFoundError(f"Required model file not found: {model_path}")
            logger.warning("Model file not found, using randomly initialized weights.")
            return

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning("Missing state dict keys during load: %d", len(missing_keys))
        if unexpected_keys:
            logger.warning("Unexpected state dict keys during load: %d", len(unexpected_keys))
        logger.info("Checkpoint loaded with strict=False")

    def _decode_video_to_frames(self, video_bytes: bytes, max_frames: int = 160) -> list[np.ndarray]:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as temp_video:
            temp_video.write(video_bytes)
            temp_video.flush()
            cap = cv2.VideoCapture(temp_video.name)

            if not cap.isOpened():
                raise ValueError("Could not decode uploaded video stream.")

            frames: list[np.ndarray] = []
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()

        if len(frames) < 16:
            raise ValueError("Insufficient frames for robust inference. Please record at least 3 seconds.")

        logger.info("Decoded %d frames from uploaded video", len(frames))
        return frames

    def _extract_face_roi(self, frame_bgr: np.ndarray, previous_box: tuple[int, int, int, int] | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            face_box = (int(x), int(y), int(w), int(h))
        elif previous_box is not None:
            face_box = previous_box
        else:
            height, width = frame_bgr.shape[:2]
            side = min(height, width)
            x = (width - side) // 2
            y = (height - side) // 2
            face_box = (x, y, side, side)

        x, y, w, h = face_box
        x2 = min(frame_bgr.shape[1], x + w)
        y2 = min(frame_bgr.shape[0], y + h)
        face_crop = frame_bgr[max(0, y):y2, max(0, x):x2]
        if face_crop.size == 0:
            face_crop = frame_bgr

        return face_crop, face_box

    def preprocess_frames(self, frames: np.ndarray | list[np.ndarray], *, input_is_rgb: bool = True) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError("Frames must have shape (T, H, W, 3).")
            frame_sequence = [frame for frame in frames]
        else:
            frame_sequence = frames

        if len(frame_sequence) < 16:
            raise ValueError("Insufficient frames for robust inference. Please record at least 3 seconds.")

        processed_frames: list[np.ndarray] = []

        previous_box: tuple[int, int, int, int] | None = None
        for frame in frame_sequence:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if input_is_rgb else frame
            face_crop, previous_box = self._extract_face_roi(frame_bgr, previous_box)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
            face_norm = face_rgb.astype(np.float32) / 255.0
            face_norm = (face_norm - 0.5) / 0.5
            processed_frames.append(face_norm)

        video_array = np.stack(processed_frames, axis=0)  # (T, H, W, C)
        video_array = np.transpose(video_array, (3, 0, 1, 2))  # (C, T, H, W)
        video_array = np.expand_dims(video_array, axis=0)  # (B, C, T, H, W)

        tensor = torch.from_numpy(video_array).to(self.device, dtype=torch.float32)
        logger.info(
            "Preprocessed tensor shape=%s on device=%s from %d frames",
            tuple(tensor.shape),
            self.device,
            len(processed_frames),
        )
        return tensor

    def preprocess_video(self, video_bytes: bytes) -> torch.Tensor:
        frames = self._decode_video_to_frames(video_bytes)
        return self.preprocess_frames(frames, input_is_rgb=False)

    def infer_raw_outputs(self, model_input: torch.Tensor) -> dict[str, list[float]]:
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model(model_input)
        latency_ms = (time.perf_counter() - start) * 1000.0
        output_np = output.squeeze(0).detach().cpu().numpy().astype(float)
        logger.info("Inference latency: %.2f ms", latency_ms)
        return {"raw_outputs": output_np.tolist()}

    def predict_metrics_from_video(self, video_bytes: bytes) -> dict[str, list[float]]:
        model_input = self.preprocess_video(video_bytes)
        return self.infer_raw_outputs(model_input)

    def predict_metrics_from_frames(self, frames_rgb: np.ndarray | list[np.ndarray]) -> dict[str, list[float]]:
        model_input = self.preprocess_frames(frames_rgb, input_is_rgb=True)
        return self.infer_raw_outputs(model_input)

    @torch.inference_mode()
    def predict_metrics_from_trace(self, rgb_trace: np.ndarray) -> dict[str, list[float]]:
        # Backward-compatible path for existing callers/tests using mean RGB trace.
        x = torch.tensor(rgb_trace, dtype=torch.float32, device=self.device).T.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return self.infer_raw_outputs(x)
