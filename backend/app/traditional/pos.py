import numpy as np


class POSProcessor:
    """Auxiliary POS stream for validation; never used for API metrics."""

    def __init__(
        self,
        window_size: int = 32,
        fps: float = 30.0,
        bandpass_low_hz: float = 0.7,
        bandpass_high_hz: float = 4.0,
    ) -> None:
        self.window_size = window_size
        self.fps = fps
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self.projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float32)

    def process(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("Expected frames_rgb shape (T, H, W, 3).")
        rgb_trace = frames_rgb.mean(axis=(1, 2))
        return self.process_trace(rgb_trace)

    def process_trace(self, rgb_trace: np.ndarray) -> np.ndarray:
        if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
            raise ValueError("Expected rgb_trace shape (T, 3).")

        rgb = rgb_trace.astype(np.float32)
        trace_len = rgb.shape[0]
        output = np.zeros(trace_len, dtype=np.float32)

        for n in range(self.window_size, trace_len + 1):
            window = rgb[n - self.window_size : n]
            normalized = window / (np.mean(window, axis=0, keepdims=True) + 1e-8)
            s = self.projection @ normalized.T
            std_ratio = np.std(s[0]) / (np.std(s[1]) + 1e-8)
            h = s[0] + std_ratio * s[1]
            h = h - np.mean(h)
            output[n - self.window_size : n] += h

        return self._bandpass_and_normalize(output)

    def _bandpass_and_normalize(self, signal: np.ndarray) -> np.ndarray:
        centered = signal - np.mean(signal)
        n = centered.shape[0]
        if n < 4:
            return centered
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        fft_values = np.fft.rfft(centered)
        mask = (freqs >= self.bandpass_low_hz) & (freqs <= self.bandpass_high_hz)
        fft_values[~mask] = 0.0
        filtered = np.fft.irfft(fft_values, n=n).astype(np.float32)
        return filtered / (np.std(filtered) + 1e-8)


def extract_pos_signal(rgb_trace: np.ndarray, window_size: int = 32) -> np.ndarray:
    """Backwards-compatible function wrapper around :class:`POSProcessor`."""
    return POSProcessor(window_size=window_size).process_trace(rgb_trace)
