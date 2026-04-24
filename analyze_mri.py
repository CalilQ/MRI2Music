import argparse
from pathlib import Path
from typing import Optional, Tuple

from mri2music.inspect_utils import (
    compute_global_statistics,
    compute_slice_statistics,
    compute_voxel_characteristics,
    load_json_metadata,
    load_nifti_image,
    plot_slice_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_mri",
        description="Analyze brain MRI data and plot slice-specific characteristics.",
    )
    parser.add_argument("--nifti", "-n", required=True, help="Path to the input NIfTI MRI file")
    parser.add_argument("--json", "-j", help="Optional metadata JSON file")
    parser.add_argument("--figs-dir", default="figs", help="Directory to save the plot")
    parser.add_argument("--plot-name", default="slice_characteristics.png", help="Name of the output plot file")
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Slice axis to analyze (default: 2)",
    )
    parser.add_argument(
        "--voxel",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Voxel coordinates to inspect (default: center voxel)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    nifti_path = Path(args.nifti)
    if not nifti_path.exists():
        print(f"Error: NIfTI file not found: {nifti_path}")
        return 1

    metadata = None
    if args.json:
        metadata_path = Path(args.json)
        if not metadata_path.exists():
            print(f"Error: JSON metadata file not found: {metadata_path}")
            return 1
        metadata = load_json_metadata(metadata_path)

    volume, image, header = load_nifti_image(nifti_path)

    global_stats = compute_global_statistics(volume)
    print("Global characteristics:")
    print(f"  Mean intensity: {global_stats['mean_intensity']:.6g}")
    print(f"  Std intensity: {global_stats['std_intensity']:.6g}")
    for label, value in global_stats["percentiles"].items():
        print(f"  {label} percentile: {value:.6g}")

    slice_stats = compute_slice_statistics(volume, axis=args.axis)
    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figs_dir / args.plot_name
    plot_slice_statistics(slice_stats, plot_path)
    print(f"Saved slice-specific plot: {plot_path}")

    voxel_index = tuple(args.voxel) if args.voxel else None
    voxel_stats = compute_voxel_characteristics(volume, index=voxel_index)
    print("Voxel characteristics:")
    print(f"  Voxel index: {voxel_stats['voxel_index']}")
    print(f"  Voxel intensity: {voxel_stats['voxel_intensity']:.6g}")
    print(f"  Normalized intensity: {voxel_stats['normalized_intensity']:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
