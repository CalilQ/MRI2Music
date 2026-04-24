from pathlib import Path

import pytest

from mri2music.inspect_utils import (
    compute_global_statistics,
    compute_slice_statistics,
    compute_voxel_characteristics,
    create_slice_gif,
    extract_nifti_characteristics,
    load_json_metadata,
    load_nifti_image,
    plot_slice_statistics,
    print_nifti_characteristics,
)


data_dir = Path("data") / "open_source"
nii_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.nii.gz"
json_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.json"


def test_load_nifti_image():
    volume, image, header = load_nifti_image(nii_path)

    assert volume.ndim == 3
    assert image is not None
    assert header is not None
    assert tuple(volume.shape) == tuple(image.shape)


def test_load_json_metadata():
    metadata = load_json_metadata(json_path)

    assert isinstance(metadata, dict)
    assert metadata.get("Modality") == "MR"
    assert metadata.get("BidsGuess") is not None


def test_extract_nifti_characteristics():
    volume, image, header = load_nifti_image(nii_path)
    metadata = load_json_metadata(json_path)
    characteristics = extract_nifti_characteristics(image, header, metadata)

    assert characteristics["dimensions"] == tuple(image.shape)[:3]
    assert "image_orientation" in characteristics
    assert characteristics["slice_thickness"] is not None
    assert characteristics["pixel_spacing"] is not None
    assert characteristics["repetition_time"] is not None
    assert characteristics["echo_time"] is not None
    assert characteristics["inversion_time"] is not None
    assert characteristics["flip_angle"] is not None
    assert characteristics["repetition_time"] == metadata.get("RepetitionTime")
    assert characteristics["echo_time"] == metadata.get("EchoTime")
    assert characteristics["inversion_time"] == metadata.get("InversionTime")
    assert characteristics["flip_angle"] == metadata.get("FlipAngle")


def test_create_slice_gif(tmp_path):
    volume, image, header = load_nifti_image(nii_path)
    output_file = tmp_path / "test_mri_slices.gif"
    gif_path = create_slice_gif(volume, output_file, axis=2, max_frames=10, duration=50)

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_print_nifti_characteristics(capsys):
    volume, image, header = load_nifti_image(nii_path)
    metadata = load_json_metadata(json_path)
    characteristics = extract_nifti_characteristics(image, header, metadata)

    print_nifti_characteristics(characteristics)
    captured = capsys.readouterr()

    assert "MRI characteristics:" in captured.out
    assert "Dimensions (x, y, z):" in captured.out
    assert "Repetition time (TR):" in captured.out


def test_compute_global_statistics():
    volume, image, header = load_nifti_image(nii_path)
    stats = compute_global_statistics(volume)

    assert stats["mean_intensity"] >= 0.0
    assert stats["std_intensity"] >= 0.0
    assert "p10" in stats["percentiles"]
    assert "p50" in stats["percentiles"]
    assert "p90" in stats["percentiles"]
    assert stats["percentiles"]["p10"] <= stats["percentiles"]["p50"] <= stats["percentiles"]["p90"]


def test_compute_slice_statistics():
    volume, image, header = load_nifti_image(nii_path)
    stats = compute_slice_statistics(volume, axis=2)

    assert stats["slice_means"].shape[0] == volume.shape[2]
    assert stats["slice_stds"].shape[0] == volume.shape[2]
    assert stats["edge_strengths"].shape[0] == volume.shape[2]
    assert (stats["slice_means"] >= 0).all()
    assert (stats["slice_stds"] >= 0).all()
    assert (stats["edge_strengths"] >= 0).all()


def test_compute_voxel_characteristics():
    volume, image, header = load_nifti_image(nii_path)
    stats = compute_voxel_characteristics(volume)

    assert isinstance(stats["voxel_index"], tuple)
    assert stats["voxel_intensity"] == float(volume[stats["voxel_index"]])
    assert 0.0 <= stats["normalized_intensity"] <= 1.0


def test_plot_slice_statistics(tmp_path):
    volume, image, header = load_nifti_image(nii_path)
    stats = compute_slice_statistics(volume, axis=2)
    plot_path = tmp_path / "slice_stats.png"
    result_path = plot_slice_statistics(stats, plot_path)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
