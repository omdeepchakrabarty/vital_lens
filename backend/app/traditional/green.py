import numpy as np


class GreenChannelProcessor:
    """Auxiliary green-channel validation stream; excluded from API metrics."""

    def __init__(self, fps: float = 30.0, bandpass_low_hz: float = 0.7, bandpass_high_hz: float = 4.0) -> None:
        self.fps = fps
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz

    def process(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("Expected frames_rgb shape (T, H, W, 3).")
        rgb_trace = frames_rgb.mean(axis=(1, 2))
        return self.process_trace(rgb_trace)

    def process_trace(self, rgb_trace: np.ndarray) -> np.ndarray:
        if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
            raise ValueError("Expected rgb_trace shape (T, 3).")
        green = rgb_trace[:, 1].astype(np.float32)
        return self._bandpass_and_normalize(green)

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


def extract_green_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """Backwards-compatible function wrapper around :class:`GreenChannelProcessor`."""
    return GreenChannelProcessor().process_trace(rgb_trace)
