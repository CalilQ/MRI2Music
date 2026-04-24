import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt


DEFAULT_SAMPLE_RATE = 44100
SAFE_MIN_FREQ = 80.0
SAFE_MAX_FREQ = 2000.0
DEFAULT_FREQ_LOW = 220.0
DEFAULT_FREQ_HIGH = 880.0
DEFAULT_PEAK = 0.95
MRI_DERIVED_WAVEFORMS = {"Slice Mean", "Slice Std", "Edge Strength", "x-sum", "y-sum", "Noise Vector"}
MRI_TEXTURE_WAVEFORMS = {"Texture Noise"}
MRI_CURVE_SOURCES = MRI_DERIVED_WAVEFORMS.union(MRI_TEXTURE_WAVEFORMS)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize_curve(curve: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(curve), dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val <= min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def mri_slice_to_noise_vector(slice_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D MRI slice into a normalized 1D noise-like vector."""
    # Convert the slice to float for processing.
    slice_array = np.asarray(slice_2d, dtype=np.float64)
    if slice_array.ndim != 2:
        raise ValueError("slice_2d must be a 2D NumPy array.")

    # Normalize the slice intensities to the range [0, 1].
    min_val = float(np.min(slice_array))
    max_val = float(np.max(slice_array))
    if max_val <= min_val:
        normalized_slice = np.zeros_like(slice_array, dtype=np.float64)
    else:
        normalized_slice = (slice_array - min_val) / (max_val - min_val)

    # Blur the normalized slice to capture the low-frequency structure.
    blurred_slice = gaussian_filter(normalized_slice, sigma=1.0)

    # Subtract the blur from the original normalized slice to get a high-pass version.
    high_pass_slice = normalized_slice - blurred_slice

    # Collapse the 2D slice into a 1D vector by averaging each row.
    noise_vector = np.mean(high_pass_slice, axis=1)

    # Center the vector around zero.
    noise_vector = noise_vector - np.mean(noise_vector)

    # Normalize the centered vector to the range [-1, 1].
    peak = float(np.max(np.abs(noise_vector)))
    if peak <= 0.0:
        return np.zeros_like(noise_vector, dtype=np.float64)
    return (noise_vector / peak).astype(np.float64)


def mri_slice_to_texture_signal(slice_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D MRI slice into a longer white-ish texture signal."""
    # Convert the slice to float so filtering and normalization are stable.
    slice_array = np.asarray(slice_2d, dtype=np.float64)
    if slice_array.ndim != 2:
        raise ValueError("slice_2d must be a 2D NumPy array.")

    # Normalize the slice intensities to the range [0, 1].
    min_val = float(np.min(slice_array))
    max_val = float(np.max(slice_array))
    if max_val <= min_val:
        normalized_slice = np.zeros_like(slice_array, dtype=np.float64)
    else:
        normalized_slice = (slice_array - min_val) / (max_val - min_val)

    # Remove broad structure so the remaining detail is more noise-like.
    blurred_slice = gaussian_filter(normalized_slice, sigma=1.0)
    high_pass_slice = normalized_slice - blurred_slice

    # Flatten with alternating row directions to avoid large jumps at row edges.
    row_segments = []
    for row_index, row in enumerate(high_pass_slice):
        row_segments.append(row if row_index % 2 == 0 else row[::-1])
    texture_signal = np.concatenate(row_segments, axis=0)

    # Emphasize rapid local changes so the signal sounds whiter.
    texture_signal = np.diff(texture_signal, prepend=texture_signal[0])

    # Center the signal around zero before normalizing.
    texture_signal = texture_signal - np.mean(texture_signal)

    # Normalize the output to the range [-1, 1].
    peak = float(np.max(np.abs(texture_signal)))
    if peak <= 0.0:
        return np.zeros_like(texture_signal, dtype=np.float64)
    return (texture_signal / peak).astype(np.float64)


def map_scalar_to_frequency(
    value: float,
    source_min: float,
    source_max: float,
    min_freq: float = DEFAULT_FREQ_LOW,
    max_freq: float = DEFAULT_FREQ_HIGH,
) -> float:
    if source_max <= source_min:
        return clamp(value, min_freq, max_freq)
    normalized = (value - source_min) / (source_max - source_min)
    normalized = clamp(normalized, 0.0, 1.0)
    return min_freq + normalized * (max_freq - min_freq)


def _make_waveform(
    waveform_type: str,
    frequency: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    sample_count = max(1, int(sample_rate * duration))
    t = np.linspace(0.0, duration, sample_count, endpoint=False, dtype=np.float64)
    phase = 2.0 * np.pi * frequency * t

    if waveform_type == "square":
        return np.sign(np.sin(phase)).astype(np.float64)
    if waveform_type == "saw":
        return (2.0 * (frequency * t - np.floor(0.5 + frequency * t))).astype(np.float64)
    if waveform_type == "triangle":
        return (2.0 * np.abs(2.0 * (frequency * t - np.floor(frequency * t + 0.5))) - 1.0).astype(np.float64)
    return np.sin(phase).astype(np.float64)


def _resample_curve(curve: np.ndarray, length: int) -> np.ndarray:
    if curve.size == 0:
        return np.zeros(length, dtype=np.float64)
    positions = np.linspace(0, curve.size - 1, length)
    original = np.arange(curve.size, dtype=np.float64)
    return np.interp(positions, original, curve).astype(np.float64)


def _generate_frequency_curve(
    frequency: float,
    modulation_curve: Optional[Iterable[float]],
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_freq: float = DEFAULT_FREQ_LOW,
    max_freq: float = DEFAULT_FREQ_HIGH,
) -> np.ndarray:
    sample_count = max(1, int(sample_rate * duration))
    if modulation_curve is None:
        return np.full(sample_count, float(frequency), dtype=np.float64)
    curve = normalize_curve(modulation_curve)
    if curve.size == 0:
        return np.full(sample_count, float(frequency), dtype=np.float64)
    curve = _resample_curve(curve, sample_count)
    instantaneous = min_freq + curve * (max_freq - min_freq)
    return np.clip(instantaneous, SAFE_MIN_FREQ, SAFE_MAX_FREQ)


def create_waveform(
    waveform_type: str,
    frequency: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude_modulation_curve: Optional[Iterable[float]] = None,
    frequency_modulation_curve: Optional[Iterable[float]] = None,
    freq_min: float = DEFAULT_FREQ_LOW,
    freq_max: float = DEFAULT_FREQ_HIGH,
) -> np.ndarray:
    sample_count = max(1, int(sample_rate * duration))
    if waveform_type in MRI_TEXTURE_WAVEFORMS and amplitude_modulation_curve is not None:
        curve_values = np.asarray(list(amplitude_modulation_curve), dtype=np.float64)
        if curve_values.size == 0:
            return np.zeros(sample_count, dtype=np.float32)
        return _resample_curve(curve_values, sample_count).astype(np.float32)

    if waveform_type in MRI_DERIVED_WAVEFORMS and amplitude_modulation_curve is not None:
        curve_values = np.asarray(list(amplitude_modulation_curve), dtype=np.float64)
        if curve_values.size == 0:
            return np.zeros(sample_count, dtype=np.float32)
        normalized_curve = normalize_curve(curve_values)
        cycle_length = max(2, int(round(sample_rate / max(frequency, 1.0))))
        positions = np.linspace(0, normalized_curve.size - 1, cycle_length)
        one_cycle = np.interp(positions, np.arange(normalized_curve.size), normalized_curve)
        one_cycle = 2.0 * one_cycle - 1.0
        repeated = np.tile(one_cycle, int(np.ceil(sample_count / cycle_length)))[:sample_count]
        return repeated.astype(np.float32)

    if frequency_modulation_curve is not None:
        frequency_array = _generate_frequency_curve(frequency, frequency_modulation_curve, duration, sample_rate, freq_min, freq_max)
        phase = 2.0 * np.pi * np.cumsum(frequency_array) / sample_rate
    else:
        sample_times = np.linspace(0.0, duration, sample_count, endpoint=False, dtype=np.float64)
        phase = 2.0 * np.pi * frequency * sample_times

    if waveform_type == "square":
        base = np.sign(np.sin(phase)).astype(np.float64)
    elif waveform_type == "saw":
        phase_cycles = phase / (2.0 * np.pi)
        base = (2.0 * (phase_cycles - np.floor(0.5 + phase_cycles))).astype(np.float64)
    elif waveform_type == "triangle":
        phase_cycles = phase / (2.0 * np.pi)
        base = (2.0 * np.abs(2.0 * (phase_cycles - np.floor(phase_cycles + 0.5))) - 1.0).astype(np.float64)
    else:
        base = np.sin(phase).astype(np.float64)

    if amplitude_modulation_curve is not None:
        curve = normalize_curve(amplitude_modulation_curve)
        curve = _resample_curve(curve, base.shape[0])
        modulation = 0.3 + 0.7 * curve
        base = base * modulation
    return base.astype(np.float32)


def create_envelope(
    curve: Optional[Iterable[float]],
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    sample_count = max(1, int(sample_rate * duration))
    if curve is None:
        envelope = np.ones(sample_count, dtype=np.float32)
    else:
        curve_array = normalize_curve(curve)
        envelope = _resample_curve(curve_array, sample_count).astype(np.float32)
        envelope = 0.1 + 0.9 * envelope
    attack = min(int(sample_rate * 0.02), sample_count)
    decay = min(int(sample_rate * 0.05), sample_count)
    if attack > 1:
        envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if decay > 1:
        envelope[-decay:] *= np.linspace(1.0, 0.0, decay, dtype=np.float32)
    return envelope


def normalize_audio(samples: np.ndarray, peak: float = DEFAULT_PEAK) -> np.ndarray:
    if samples.size == 0:
        return samples
    absolute = np.max(np.abs(samples))
    if absolute <= 0.0:
        return samples
    return (samples / absolute) * peak


def apply_saturation(
    samples: np.ndarray,
    saturation_curve: Optional[Iterable[float]] = None,
    amount: float = 1.0,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or saturation_curve is None or amount <= 0.0:
        return samples

    curve = normalize_curve(saturation_curve)
    if curve.size == 0:
        return samples

    # Map the MRI curve to a time-varying drive so denser regions saturate harder.
    drive = 1.0 + (2.0 + 6.0 * max(0.0, amount)) * _resample_curve(curve, samples.shape[0])
    saturated = np.tanh(samples.astype(np.float64) * drive)
    return saturated.astype(np.float32)


def apply_amplitude_dependent_distortion(
    samples: np.ndarray,
    amount: float = 0.0,
    envelope_smoothing_hz: float = 30.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or amount <= 0.0:
        return samples

    # Follow the signal amplitude with a simple one-pole envelope follower.
    smoothing_hz = clamp(float(envelope_smoothing_hz), 1.0, sample_rate * 0.25)
    alpha = math.exp(-2.0 * math.pi * smoothing_hz / sample_rate)
    envelope = np.empty(samples.shape[0], dtype=np.float64)
    previous = 0.0
    for index, sample in enumerate(samples.astype(np.float64)):
        level = abs(sample)
        previous = alpha * previous + (1.0 - alpha) * level
        envelope[index] = previous

    envelope_peak = float(np.max(envelope))
    if envelope_peak > 0.0:
        envelope /= envelope_peak

    # Louder regions receive more drive, so harmonics bloom with the envelope.
    drive = 1.0 + (2.0 + 8.0 * max(0.0, amount)) * envelope
    distorted = np.tanh(samples.astype(np.float64) * drive)
    return distorted.astype(np.float32)


def apply_filter(
    samples: np.ndarray,
    filter_type: str = "none",
    cutoff_low: Optional[float] = None,
    cutoff_high: Optional[float] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples

    normalized_type = filter_type.strip().lower()
    nyquist = max(sample_rate * 0.5, 1.0)

    if normalized_type == "low pass":
        if cutoff_high is None:
            return samples
        cutoff = clamp(float(cutoff_high), 1.0, nyquist * 0.99)
        sos = butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    elif normalized_type == "high pass":
        if cutoff_low is None:
            return samples
        cutoff = clamp(float(cutoff_low), 1.0, nyquist * 0.99)
        sos = butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
    elif normalized_type == "band pass":
        if cutoff_low is None or cutoff_high is None:
            return samples
        low = clamp(float(cutoff_low), 1.0, nyquist * 0.98)
        high = clamp(float(cutoff_high), low + 1.0, nyquist * 0.99)
        if high <= low:
            return samples
        sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    else:
        return samples

    filtered = sosfiltfilt(sos, samples.astype(np.float64))
    return filtered.astype(np.float32)


def synthesize_sample(
    sample_type: str,
    waveform_type: str,
    frequency: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    waveform_curve: Optional[Iterable[float]] = None,
    envelope_curve: Optional[Iterable[float]] = None,
    frequency_modulation_curve: Optional[Iterable[float]] = None,
    saturation_curve: Optional[Iterable[float]] = None,
    saturation_amount: float = 1.0,
    amplitude_distortion_amount: float = 0.0,
    filter_type: str = "none",
    filter_cutoff_low: Optional[float] = None,
    filter_cutoff_high: Optional[float] = None,
    freq_min: float = DEFAULT_FREQ_LOW,
    freq_max: float = DEFAULT_FREQ_HIGH,
) -> np.ndarray:
    waveform = create_waveform(
        waveform_type,
        frequency,
        duration,
        sample_rate,
        amplitude_modulation_curve=waveform_curve,
        frequency_modulation_curve=frequency_modulation_curve,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    envelope = create_envelope(envelope_curve, duration, sample_rate)
    sample = (waveform * envelope).astype(np.float32)
    sample = apply_saturation(sample, saturation_curve=saturation_curve, amount=saturation_amount)
    sample = apply_amplitude_dependent_distortion(
        sample,
        amount=amplitude_distortion_amount,
        sample_rate=sample_rate,
    )
    sample = apply_filter(
        sample,
        filter_type=filter_type,
        cutoff_low=filter_cutoff_low,
        cutoff_high=filter_cutoff_high,
        sample_rate=sample_rate,
    )
    return normalize_audio(sample).astype(np.float32)


def save_wav(path: Path, samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.float32)
    samples = normalize_audio(samples)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def save_sample_metadata(path: Path, metadata: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


class SoundDevicePlayback:
    def stop(self) -> None:
        import sounddevice as sd
        sd.stop()

    def close(self) -> None:
        return None


def preview_audio(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError("Preview requires sounddevice. Install it with pip install sounddevice") from exc

    samples = np.asarray(samples, dtype=np.float32)
    samples = normalize_audio(samples)
    sd.play(samples, samplerate=sample_rate, blocking=False)
    return SoundDevicePlayback()
