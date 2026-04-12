import numpy as np


def extract_pos_signal(rgb_trace: np.ndarray, window_size: int = 32) -> np.ndarray:
    """Plane-Orthogonal-to-Skin method baseline implementation."""
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("Expected rgb_trace shape (T, 3).")

    rgb = rgb_trace.astype(np.float32)
    trace_len = rgb.shape[0]
    output = np.zeros(trace_len, dtype=np.float32)

    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float32)

    for n in range(window_size, trace_len + 1):
        window = rgb[n - window_size : n]
        normalized = window / (np.mean(window, axis=0, keepdims=True) + 1e-8)
        s = projection @ normalized.T
        std_ratio = np.std(s[0]) / (np.std(s[1]) + 1e-8)
        h = s[0] + std_ratio * s[1]
        h = h - np.mean(h)
        output[n - window_size : n] += h

    output = output - np.mean(output)
    return output / (np.std(output) + 1e-8)
