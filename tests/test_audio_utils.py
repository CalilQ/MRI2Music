from pathlib import Path

import numpy as np

from mri2music.audio_utils import (
    apply_filter,
    apply_amplitude_dependent_distortion,
    DEFAULT_SAMPLE_RATE,
    apply_saturation,
    clamp,
    create_envelope,
    create_waveform,
    map_scalar_to_frequency,
    mri_slice_to_noise_vector,
    mri_slice_to_texture_signal,
    normalize_audio,
    save_wav,
    save_sample_metadata,
    synthesize_sample,
)


def test_map_scalar_to_frequency():
    freq = map_scalar_to_frequency(0.5, 0.0, 1.0, 220.0, 880.0)
    assert 220.0 <= freq <= 880.0


def test_create_waveform():
    wave = create_waveform("sine", 440.0, 0.5, sample_rate=22050)
    assert wave.shape[0] == 11025
    assert np.max(np.abs(wave)) <= 1.0


def test_mri_slice_to_noise_vector():
    slice_2d = np.arange(64, dtype=np.float64).reshape(8, 8)
    vector = mri_slice_to_noise_vector(slice_2d)
    assert vector.shape == (8,)
    assert np.isclose(np.mean(vector), 0.0, atol=1e-7)
    assert np.max(np.abs(vector)) <= 1.0


def test_mri_slice_to_texture_signal():
    slice_2d = np.arange(16, dtype=np.float64).reshape(4, 4)
    signal = mri_slice_to_texture_signal(slice_2d)
    assert signal.shape == (16,)
    assert np.isclose(np.mean(signal), 0.0, atol=1e-7)
    assert np.max(np.abs(signal)) <= 1.0


def test_create_envelope():
    envelope = create_envelope([0.0, 0.5, 1.0], 1.0, sample_rate=22050)
    assert envelope.shape[0] == 22050
    assert np.all(envelope >= 0.0)
    assert np.all(envelope <= 1.0)


def test_normalize_audio():
    array = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    normalized = normalize_audio(array, peak=0.5)
    assert np.isclose(np.max(np.abs(normalized)), 0.5)


def test_apply_saturation_changes_signal():
    samples = np.linspace(-0.8, 0.8, 16, dtype=np.float32)
    saturated = apply_saturation(samples, saturation_curve=[0.0, 0.5, 1.0], amount=1.0)
    assert saturated.shape == samples.shape
    assert np.max(np.abs(saturated)) <= 1.0 + 1e-6
    assert not np.allclose(saturated, samples)


def test_apply_filter_changes_signal():
    t = np.linspace(0.0, 0.25, 5512, endpoint=False, dtype=np.float64)
    samples = (0.7 * np.sin(2.0 * np.pi * 220.0 * t) + 0.3 * np.sin(2.0 * np.pi * 4000.0 * t)).astype(np.float32)
    filtered = apply_filter(samples, filter_type="Low pass", cutoff_high=600.0, sample_rate=22050)
    assert filtered.shape == samples.shape
    assert not np.allclose(filtered, samples)


def test_apply_amplitude_dependent_distortion_changes_signal():
    t = np.linspace(0.0, 0.25, 5512, endpoint=False, dtype=np.float64)
    envelope = np.linspace(0.1, 1.0, t.shape[0], dtype=np.float64)
    samples = (envelope * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    distorted = apply_amplitude_dependent_distortion(samples, amount=1.0, sample_rate=22050)
    assert distorted.shape == samples.shape
    assert np.max(np.abs(distorted)) <= 1.0 + 1e-6
    assert not np.allclose(distorted, samples)


def test_synthesize_sample_and_save(tmp_path):
    sample = synthesize_sample(
        sample_type="tone",
        waveform_type="sine",
        frequency=440.0,
        duration=0.5,
        sample_rate=22050,
        waveform_curve=[0.0, 0.5, 1.0],
        envelope_curve=[1.0, 0.0, 0.5],
    )
    assert sample.shape[0] == 11025
    output_wav = tmp_path / "test.wav"
    save_wav(output_wav, sample, sample_rate=22050)
    assert output_wav.exists()
    metadata_path = tmp_path / "test.json"
    save_sample_metadata(metadata_path, {"sample_file": "test.wav"})
    assert metadata_path.exists()


def test_synthesize_sample_with_noise_vector_curve():
    slice_2d = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 3.0],
        ],
        dtype=np.float64,
    )
    noise_vector = mri_slice_to_noise_vector(slice_2d)
    sample = synthesize_sample(
        sample_type="tone",
        waveform_type="Noise Vector",
        frequency=330.0,
        duration=0.25,
        sample_rate=22050,
        waveform_curve=noise_vector,
        envelope_curve=noise_vector,
        frequency_modulation_curve=noise_vector,
        freq_min=220.0,
        freq_max=660.0,
    )
    assert sample.shape[0] == 5512
    assert np.max(np.abs(sample)) <= 0.95 + 1e-6
    assert not np.allclose(sample, 0.0)


def test_create_waveform_with_texture_noise_curve():
    slice_2d = np.arange(25, dtype=np.float64).reshape(5, 5)
    texture_signal = mri_slice_to_texture_signal(slice_2d)
    sample = create_waveform(
        "Texture Noise",
        330.0,
        0.25,
        sample_rate=22050,
        amplitude_modulation_curve=texture_signal,
    )
    assert sample.shape[0] == 5512
    assert np.max(np.abs(sample)) <= 1.0 + 1e-6
    assert not np.allclose(sample, 0.0)


def test_synthesize_sample_with_saturation_curve():
    slice_2d = np.arange(25, dtype=np.float64).reshape(5, 5)
    texture_signal = mri_slice_to_texture_signal(slice_2d)
    sample = synthesize_sample(
        sample_type="tone",
        waveform_type="sine",
        frequency=220.0,
        duration=0.25,
        sample_rate=22050,
        saturation_curve=texture_signal,
    )
    assert sample.shape[0] == 5512
    assert np.max(np.abs(sample)) <= 0.95 + 1e-6
    assert not np.allclose(sample, 0.0)


def test_synthesize_sample_with_bandpass_filter():
    sample = synthesize_sample(
        sample_type="tone",
        waveform_type="saw",
        frequency=440.0,
        duration=0.25,
        sample_rate=22050,
        filter_type="Band pass",
        filter_cutoff_low=300.0,
        filter_cutoff_high=1500.0,
    )
    assert sample.shape[0] == 5512
    assert np.max(np.abs(sample)) <= 0.95 + 1e-6
    assert not np.allclose(sample, 0.0)


def test_synthesize_sample_with_amplitude_dependent_distortion():
    sample = synthesize_sample(
        sample_type="tone",
        waveform_type="sine",
        frequency=220.0,
        duration=0.25,
        sample_rate=22050,
        amplitude_distortion_amount=1.0,
    )
    assert sample.shape[0] == 5512
    assert np.max(np.abs(sample)) <= 0.95 + 1e-6
    assert not np.allclose(sample, 0.0)
