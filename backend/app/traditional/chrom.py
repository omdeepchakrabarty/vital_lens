import numpy as np


def extract_chrom_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """CHROM method as a traditional rPPG baseline."""
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError("Expected rgb_trace shape (T, 3).")

    rgb = rgb_trace.astype(np.float32)
    rgb = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)
    x_comp = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y_comp = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]

    alpha = np.std(x_comp) / (np.std(y_comp) + 1e-8)
    s = x_comp - alpha * y_comp
    s = s - np.mean(s)
    return s / (np.std(s) + 1e-8)
