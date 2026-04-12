from io import BytesIO

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
    raw = np.frombuffer(video_bytes, dtype=np.uint8)
    cap = cv2.VideoCapture()
    if not cap.open(BytesIO(raw).read(), cv2.CAP_FFMPEG):
        temp_path = "/tmp/upload.webm"
        with open(temp_path, "wb") as file:
            file.write(video_bytes)
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

    cap.release()

    if len(frames_rgb) < 16:
        raise ValueError("Insufficient frames for robust inference. Please record at least 3 seconds.")

    frames = np.stack(frames_rgb).astype(np.float32)
    trace = frames.mean(axis=(1, 2)).astype(np.float32)
    return frames, trace


def decode_video_to_rgb_trace(video_bytes: bytes, max_frames: int = 256) -> np.ndarray:
    """Compatibility helper returning only RGB trace."""
    _, trace = decode_video_to_frames_and_trace(video_bytes, max_frames=max_frames)
    return trace
