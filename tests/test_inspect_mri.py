import sys
from pathlib import Path

import inspect_mri


data_dir = Path("data") / "open_source"
nii_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.nii.gz"
json_path = data_dir / "sub-01_ses-mri_acq-mprage_T1w.json"


def test_inspect_mri_main_creates_gif(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "inspect_mri",
        "--nifti",
        str(nii_path),
        "--json",
        str(json_path),
        "--figs-dir",
        str(tmp_path),
        "--gif-name",
        "test.gif",
    ])

    status = inspect_mri.main()

    captured = capsys.readouterr()
    output_file = tmp_path / "test.gif"

    assert status == 0
    assert "MRI characteristics:" in captured.out
    assert "Saved animated GIF:" in captured.out
    assert output_file.exists()
    assert output_file.stat().st_size > 0
