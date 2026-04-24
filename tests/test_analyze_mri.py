import sys
from pathlib import Path

import analyze_mri


data_dir = Path("data") / "open_source"
nii_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.nii.gz"
json_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.json"


def test_analyze_mri_main_generates_plot(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "analyze_mri",
        "--nifti",
        str(nii_path),
        "--json",
        str(json_path),
        "--figs-dir",
        str(tmp_path),
        "--plot-name",
        "slice_characteristics.png",
    ])

    status = analyze_mri.main()
    captured = capsys.readouterr()
    plot_file = tmp_path / "slice_characteristics.png"

    assert status == 0
    assert "Global characteristics:" in captured.out
    assert "Voxel characteristics:" in captured.out
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0
