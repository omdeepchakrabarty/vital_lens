import os
import tempfile

import cv2
import numpy as np


def decode_video_to_frames_and_trace(video_bytes: bytes, max_frames: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode uploaded video and return:
    - frames_rgb: dense RGB frames in shape (T, H, W, 3)
    - rgb_trace: mean RGB trace in shape (T, 3)

    The frame stack is used by auxiliary traditional processors (POS/CHROM/Green),
    while the trace is used for DL model inference.
    """
    temp_path = ""
    cap = None

    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".webm", delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Could not decode video stream.")

        frames_rgb = []
        while len(frames_rgb) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)

    except OSError as exc:
        raise ValueError("Failed to process uploaded video.") from exc
    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if len(frames_rgb) < 16:
        raise ValueError("Insufficient frames for robust inference. Please record at least 3 seconds.")

    frames = np.stack(frames_rgb).astype(np.float32)
    trace = frames.mean(axis=(1, 2)).astype(np.float32)
    return frames, trace


def decode_video_to_rgb_trace(video_bytes: bytes, max_frames: int = 256) -> np.ndarray:
    """Compatibility helper returning only RGB trace."""
    _, trace = decode_video_to_frames_and_trace(video_bytes, max_frames=max_frames)
    return trace
