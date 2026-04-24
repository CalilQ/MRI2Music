# MRI2Music

MRI2Music is a Python tool that converts brain MRI data into configurable electronic music. It is designed to work with open source MRI data and optional audio samples, allowing flexible use of different MRI characteristics to shape the generated sound.

## Project Goal

- Read brain MRI data and metadata
- Map MRI features to audio parameters
- Generate electronic music from brain scans
- Support open source data and configurable processing modes
- Allow optional blending with open source waveform samples

## Features

- Flexible MRI characteristic mapping modes, including slice-based, histogram-based, gradient-based, and metadata-driven approaches
- Supports brain MRI input in open formats such as NIfTI and JSON metadata files
- Generates WAV audio output using configurable frequency and segment duration settings
- Optional audio sample blending for richer electronic textures
- Designed for research-driven creative audio exploration using only open source sources

## Usage

1. Provide a brain MRI file or companion metadata JSON file.
2. Configure the processing mode and audio generation parameters.
3. Generate a WAV file that reflects MRI-derived features as electronic music.

Example:

```bash
python inspect_mri.py \
  --nifti data/open_source/sub-01_ses-mri_acq-mprage_T1w.nii.gz \
  --json data/open_source/sub-01_ses-mri_acq-mprage_T1w.json
```

Then use the generated GIF and metadata insights to guide audio synthesis.

## Notes

- Focus is on brain MRI data, with the option to use open source 3D scan metadata and clinical imaging fields.
- The tool aims to remain flexible and easy to configure so different MRI characteristics can influence the resulting audio.
- Open source sample audio can be used to blend with the generated music without relying on proprietary datasets.
