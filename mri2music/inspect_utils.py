import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from PIL import Image


def load_nifti_image(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image, nib.Nifti1Header]:
    path = Path(path)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    header = img.header
    return data, img, header


def load_json_metadata(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_keys(data: Any) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            lower_key = key.lower()
            if lower_key not in flattened:
                flattened[lower_key] = value
            child_flat = _flatten_keys(value)
            for child_key, child_value in child_flat.items():
                if child_key not in flattened:
                    flattened[child_key] = child_value
    elif isinstance(data, list):
        for value in data:
            child_flat = _flatten_keys(value)
            for child_key, child_value in child_flat.items():
                if child_key not in flattened:
                    flattened[child_key] = child_value
    return flattened


def find_metadata_value(metadata: Dict[str, Any], names: List[str]) -> Optional[Any]:
    flattened = _flatten_keys(metadata)
    for name in names:
        value = flattened.get(name.lower())
        if value is not None:
            return value
    return None


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def extract_nifti_characteristics(
    img: nib.Nifti1Image,
    header: nib.Nifti1Header,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    shape = tuple(img.shape)
    zooms = tuple(header.get_zooms())
    dims = shape[:3] if len(shape) >= 3 else shape
    orientation = " ".join(nib.aff2axcodes(img.affine))
    slice_thickness = zooms[2] if len(zooms) >= 3 else None
    pixel_spacing = tuple(zooms[:2]) if len(zooms) >= 2 else None

    header_values = {
        "repetition_time": header.get("pixdim", [None, None, None, None, None])[4],
        "echo_time": header.get("te", None),
        "inversion_time": header.get("ti", None),
        "flip_angle": header.get("flip_angle", None),
    }

    characteristics = {
        "dimensions": dims,
        "image_orientation": orientation,
        "slice_thickness": slice_thickness,
        "pixel_spacing": pixel_spacing,
        "repetition_time": find_metadata_value(metadata, ["RepetitionTime", "TR", "Repetition Time"]) or header_values["repetition_time"],
        "echo_time": find_metadata_value(metadata, ["EchoTime", "TE", "Echo Time"]) or header_values["echo_time"],
        "inversion_time": find_metadata_value(metadata, ["InversionTime", "TI", "Inversion Time"]) or header_values["inversion_time"],
        "flip_angle": find_metadata_value(metadata, ["FlipAngle", "Flip Angle"]) or header_values["flip_angle"],
    }

    return characteristics


def print_nifti_characteristics(characteristics: Dict[str, Any]) -> None:
    print("MRI characteristics:")
    dims = tuple(characteristics.get("dimensions", []))
    if len(dims) >= 3:
        print(f"  Dimensions (x, y, z): {dims[0]} x {dims[1]} x {dims[2]}")
    else:
        print(f"  Dimensions: {dims}")
    print(f"  Image orientation: {characteristics['image_orientation']}")
    print(f"  Slice thickness: {_format_float(characteristics['slice_thickness'])}")
    pixel_spacing = characteristics.get("pixel_spacing")
    if pixel_spacing is not None:
        print(f"  Pixel spacing: ({_format_float(pixel_spacing[0])}, {_format_float(pixel_spacing[1])})")
    else:
        print("  Pixel spacing: N/A")
    print(f"  Repetition time (TR): {_format_float(characteristics['repetition_time'])}")
    print(f"  Echo time (TE): {_format_float(characteristics['echo_time'])}")
    print(f"  Inversion time (TI): {_format_float(characteristics['inversion_time'])}")
    print(f"  Flip angle: {_format_float(characteristics['flip_angle'])}")


def create_slice_gif(
    volume: np.ndarray,
    output_path: Path,
    axis: int = 2,
    max_frames: int = 60,
    duration: int = 80,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if volume.ndim < 2:
        raise ValueError("Volume must have at least 2 dimensions to create a GIF.")

    axis = min(max(axis, 0), volume.ndim - 1)
    num_slices = volume.shape[axis]
    indices = np.linspace(0, num_slices - 1, min(num_slices, max_frames), dtype=int)

    frames: List[Image.Image] = []
    for idx in indices:
        slice_data = np.take(volume, idx, axis=axis)
        if slice_data.ndim > 2:
            slice_data = np.mean(slice_data, axis=-1)
        slice_data = np.asarray(slice_data, dtype=np.float32)
        slice_norm = slice_data - slice_data.min()
        if slice_norm.max() > 0:
            slice_norm /= slice_norm.max()
        slice_pixels = np.uint8(np.round(slice_norm * 255.0))
        frames.append(Image.fromarray(slice_pixels, mode="L"))

    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    return output_path
