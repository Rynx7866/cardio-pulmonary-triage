
"""
ECG preprocessing utilities for PTB-XL.

Input shape:  (time_steps, 12)
Output shape: (12, time_steps)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def validate_ecg(signal):
    """Validate an ECG recording."""
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError("ECG must be a 2D array.")

    if signal.shape[1] != 12:
        raise ValueError(
            f"Expected 12 ECG leads, got {signal.shape[1]}."
        )

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains NaN or infinite values.")

    return signal


def bandpass_filter(
    signal,
    sampling_rate=100,
    lowcut=0.5,
    highcut=45.0,
    order=4,
):
    """Apply a zero-phase Butterworth bandpass filter."""
    signal = validate_ecg(signal)

    nyquist = sampling_rate / 2

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            "Cutoffs must satisfy 0 < lowcut < highcut < Nyquist."
        )

    if order < 1:
        raise ValueError("Filter order must be positive.")

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    filtered = sosfiltfilt(sos, signal, axis=0)

    return filtered.astype(np.float32)


def to_channels_first(signal):
    """Convert (time, leads) to (leads, time)."""
    signal = validate_ecg(signal)
    return np.ascontiguousarray(signal.T, dtype=np.float32)


def preprocess_ecg(
    signal,
    sampling_rate=100,
    lowcut=0.5,
    highcut=45.0,
):
    """Filter and convert one ECG recording for a 1D CNN."""
    signal = bandpass_filter(
        signal,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
    )

    signal = to_channels_first(signal)

    if not np.isfinite(signal).all():
        raise ValueError("Preprocessing produced invalid values.")

    return signal
