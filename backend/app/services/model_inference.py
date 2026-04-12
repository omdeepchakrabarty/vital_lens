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
import torch.nn.functional as F

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class BigSmallMultitaskNet(nn.Module):
    """BP4D BigSmall multitask model aligned to checkpoint parameter names."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Big branch
        self.big_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.big_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.big_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.big_conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.big_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.big_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Small branch
        self.small_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.small_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.small_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.small_conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Task heads
        self.au_fc1 = nn.Linear(64 * 9 * 9, 128)
        self.au_fc2 = nn.Linear(128, 12)

        self.bvp_fc1 = nn.Linear(64 * 9 * 9, 128)
        self.bvp_fc2 = nn.Linear(128, 1)

        self.resp_fc1 = nn.Linear(64 * 9 * 9, 128)
        self.resp_fc2 = nn.Linear(128, 1)

    @staticmethod
    def _block(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        return F.relu(conv(x), inplace=True)

    def _big_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self._block(x, self.big_conv1)
        x = F.max_pool2d(x, kernel_size=2)  # 72 -> 36
        x = self._block(x, self.big_conv2)
        x = self._block(x, self.big_conv3)
        x = F.max_pool2d(x, kernel_size=2)  # 36 -> 18
        x = self._block(x, self.big_conv4)
        x = self._block(x, self.big_conv5)
        x = self._block(x, self.big_conv6)
        x = F.max_pool2d(x, kernel_size=2)  # 18 -> 9
        return x

    def _small_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self._block(x, self.small_conv1)
        x = F.max_pool2d(x, kernel_size=2)  # 36 -> 18
        x = self._block(x, self.small_conv2)
        x = self._block(x, self.small_conv3)
        x = F.max_pool2d(x, kernel_size=2)  # 18 -> 9
        x = self._block(x, self.small_conv4)
        return x

    def forward(self, big_frame: torch.Tensor, small_frame: torch.Tensor) -> dict[str, torch.Tensor]:
        big_features = self._big_branch(big_frame)
        small_features = self._small_branch(small_frame)

        fused = big_features + small_features
        fused = fused.flatten(start_dim=1)

        au = self.au_fc2(F.relu(self.au_fc1(fused), inplace=True))
        bvp = self.bvp_fc2(F.relu(self.bvp_fc1(fused), inplace=True))
        resp = self.resp_fc2(F.relu(self.resp_fc1(fused), inplace=True))
        return {"au": au, "bvp": bvp, "resp": resp}


class ModelInferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.model_device)
        self.model = BigSmallMultitaskNet().to(self.device)
        self.input_size = (72, 72)
        self.small_input_size = (36, 36)
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
            state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
        else:
            state_dict = checkpoint

        normalized_state = {}
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("module.")
            normalized_state[normalized_key] = value

        self.model.load_state_dict(normalized_state, strict=True)
        logger.info("Checkpoint loaded with strict=True (%d tensors)", len(normalized_state))

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
        temporal_frame = np.mean(video_array, axis=0)  # (H, W, C)

        big = np.transpose(temporal_frame, (2, 0, 1))[None, ...]  # (1, C, 72, 72)
        small_hw = cv2.resize(temporal_frame, self.small_input_size, interpolation=cv2.INTER_AREA)
        small = np.transpose(small_hw, (2, 0, 1))[None, ...]  # (1, C, 36, 36)

        big_tensor = torch.from_numpy(big).to(self.device, dtype=torch.float32)
        small_tensor = torch.from_numpy(small).to(self.device, dtype=torch.float32)
        logger.info(
            "Preprocessed tensors big=%s small=%s on device=%s from %d frames",
            tuple(big_tensor.shape),
            tuple(small_tensor.shape),
            self.device,
            len(processed_frames),
        )
        return big_tensor, small_tensor

    def preprocess_video(self, video_bytes: bytes) -> torch.Tensor:
        frames = self._decode_video_to_frames(video_bytes)
        return self.preprocess_frames(frames, input_is_rgb=False)

    def infer_raw_outputs(self, model_input: torch.Tensor) -> dict[str, list[float]]:
        self.model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            multitask_output = self.model(*model_input)
        latency_ms = (time.perf_counter() - start) * 1000.0

        au = multitask_output["au"].squeeze(0).detach().cpu().numpy().astype(float)
        bvp = float(multitask_output["bvp"].squeeze().detach().cpu().item())
        resp = float(multitask_output["resp"].squeeze().detach().cpu().item())

        # keep downstream contract of four outputs while preserving multitask signal
        compact_output = [bvp, resp, float(np.mean(au)), float(np.std(au))]

        logger.info("Inference latency: %.2f ms", latency_ms)
        return {"raw_outputs": compact_output}

    def predict_metrics_from_video(self, video_bytes: bytes) -> dict[str, list[float]]:
        model_input = self.preprocess_video(video_bytes)
        return self.infer_raw_outputs(model_input)

    def predict_metrics_from_frames(self, frames_rgb: np.ndarray | list[np.ndarray]) -> dict[str, list[float]]:
        model_input = self.preprocess_frames(frames_rgb, input_is_rgb=True)
        return self.infer_raw_outputs(model_input)

    @torch.inference_mode()
    def predict_metrics_from_trace(self, rgb_trace: np.ndarray) -> dict[str, list[float]]:
        # Backward-compatible path for existing callers/tests using mean RGB trace.
        averaged = np.mean(rgb_trace, axis=0).astype(np.float32)
        repeated = np.tile(averaged[None, None, :], (72, 72, 1))
        normalized = (repeated - 0.5) / 0.5

        big = np.transpose(normalized, (2, 0, 1))[None, ...]
        small_hw = cv2.resize(normalized, self.small_input_size, interpolation=cv2.INTER_AREA)
        small = np.transpose(small_hw, (2, 0, 1))[None, ...]

        model_input = (
            torch.from_numpy(big).to(self.device, dtype=torch.float32),
            torch.from_numpy(small).to(self.device, dtype=torch.float32),
        )
        return self.infer_raw_outputs(model_input)
