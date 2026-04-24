import argparse
from pathlib import Path

from mri2music.inspect_utils import (
    create_slice_gif,
    extract_nifti_characteristics,
    load_json_metadata,
    load_nifti_image,
    print_nifti_characteristics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_mri",
        description="Inspect brain MRI data and create a slice animation gif.",
    )
    parser.add_argument("--nifti", "-n", required=True, help="Path to the input NIfTI MRI file")
    parser.add_argument("--json", "-j", help="Optional metadata JSON file to extract imaging parameters")
    parser.add_argument("--figs-dir", default="figs", help="Directory to save the animated GIF")
    parser.add_argument("--gif-name", default="mri_slices.gif", help="Name of the output GIF file")
    parser.add_argument("--axis", type=int, default=2, help="Slice axis to animate (default: 2)")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum number of GIF frames")
    parser.add_argument("--duration", type=int, default=80, help="Frame duration in milliseconds")
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
    characteristics = extract_nifti_characteristics(image, header, metadata)
    print_nifti_characteristics(characteristics)

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    output_path = figs_dir / args.gif_name
    gif_path = create_slice_gif(
        volume,
        output_path,
        axis=args.axis,
        max_frames=args.max_frames,
        duration=args.duration,
    )
    print(f"Saved animated GIF: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
