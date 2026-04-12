from io import BytesIO

import cv2
import numpy as np


def decode_video_to_rgb_trace(video_bytes: bytes, max_frames: int = 256) -> np.ndarray:
    """
    Decode uploaded video and return mean RGB trace (T, 3).
    Each frame is converted BGR->RGB and spatially pooled.
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

    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(np.mean(rgb, axis=(0, 1)))

    cap.release()

    if len(frames) < 16:
        raise ValueError("Insufficient frames for robust inference. Please record at least 3 seconds.")

    trace = np.stack(frames).astype(np.float32)
    return trace
