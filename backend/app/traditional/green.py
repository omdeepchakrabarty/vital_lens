import numpy as np


def extract_green_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """Traditional baseline method: use normalized green channel only."""
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("Expected rgb_trace shape (T, 3).")

    green = rgb_trace[:, 1].astype(np.float32)
    green = green - np.mean(green)
    std = np.std(green) + 1e-8
    return green / std
