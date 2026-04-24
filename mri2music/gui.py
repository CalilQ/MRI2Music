import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .audio_utils import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_FREQ_HIGH,
    DEFAULT_FREQ_LOW,
    MRI_CURVE_SOURCES,
    MRI_DERIVED_WAVEFORMS,
    MRI_TEXTURE_WAVEFORMS,
    map_scalar_to_frequency,
    mri_slice_to_noise_vector,
    mri_slice_to_texture_signal,
    normalize_curve,
    preview_audio,
    save_sample_metadata,
    save_wav,
    synthesize_sample,
)
from .inspect_utils import (
    compute_global_statistics,
    compute_slice_statistics,
    compute_voxel_characteristics,
    extract_nifti_characteristics,
    load_json_metadata,
    load_nifti_image,
    print_nifti_characteristics,
)


def _format_stat(value: float) -> str:
    try:
        return f"{value:.4g}"
    except (TypeError, ValueError):
        return str(value)


class MRI2MusicApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("MRI2Music Audio Mapper")
        self.root.state("zoomed")
        self.volume = None
        self.metadata = None
        self.global_stats = None
        self.slice_stats = None
        self.voxel_stats = None
        self.preview_handle = None
        self.preview_timer = None
        self.sample_output_dir = Path("samples")

        self.waveform_type_var = tk.StringVar(value="sine")
        self.waveform_slice_index_var = tk.StringVar(value="center")
        self.envelope_curve_var = tk.StringVar(value="Slice Std")
        self.envelope_slice_index_var = tk.StringVar(value="center")
        self.frequency_source_var = tk.StringVar(value="Global Mean")
        self.frequency_modulation_var = tk.StringVar(value="None")
        self.frequency_modulation_slice_index_var = tk.StringVar(value="center")
        self.saturation_source_var = tk.StringVar(value="None")
        self.saturation_slice_index_var = tk.StringVar(value="center")
        self.filter_type_var = tk.StringVar(value="None")
        self.filter_cutoff_low_var = tk.StringVar(value="400")
        self.filter_cutoff_high_var = tk.StringVar(value="2000")
        self.amplitude_distortion_amount_var = tk.StringVar(value="0.0")
        self.sample_type_var = tk.StringVar(value="tone")
        self.freq_min_var = tk.StringVar(value="220")
        self.freq_max_var = tk.StringVar(value="880")
        self.duration_var = tk.StringVar(value="2.0")
        self.sample_name_var = tk.StringVar(value="mri_sample")
        self.voxel_index_var = tk.StringVar(value="center")
        self.sample_plot_mode_var = tk.StringVar(value="few_cycles")
        self.slice_image_size = 240
        self._blank_slice_image = self._create_blank_slice_image(self.slice_image_size)

        self._build_ui()
        for var in (
            self.waveform_type_var,
            self.waveform_slice_index_var,
            self.envelope_curve_var,
            self.envelope_slice_index_var,
            self.frequency_source_var,
            self.frequency_modulation_var,
            self.frequency_modulation_slice_index_var,
            self.saturation_source_var,
            self.saturation_slice_index_var,
            self.filter_type_var,
            self.filter_cutoff_low_var,
            self.filter_cutoff_high_var,
            self.amplitude_distortion_amount_var,
            self.sample_type_var,
            self.freq_min_var,
            self.freq_max_var,
            self.duration_var,
            self.sample_plot_mode_var,
        ):
            var.trace_add("write", self._on_plot_settings_changed)
        self._update_voxel_index_visibility()
        self._update_sum_slice_visibility()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        folder_frame = tk.Frame(frame)
        folder_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        tk.Button(folder_frame, text="Select Input Folder", command=self._on_select_folder).grid(row=0, column=0, sticky="w")
        self.folder_label = tk.Label(folder_frame, text="No folder selected", anchor="w")
        self.folder_label.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        display_frame = tk.Frame(frame)
        display_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)
        display_frame.columnconfigure(3, weight=1)
        display_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(display_frame, height=12, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.status_text.configure(state=tk.DISABLED)

        waveform_slice_frame = tk.LabelFrame(display_frame, text="Waveform slice", padx=5, pady=5)
        waveform_slice_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        waveform_slice_frame.rowconfigure(0, weight=1)
        waveform_slice_frame.columnconfigure(0, weight=1)
        waveform_slice_frame.configure(width=self.slice_image_size, height=self.slice_image_size)
        waveform_slice_frame.grid_propagate(False)
        self.waveform_slice_image_label = tk.Label(waveform_slice_frame, image=self._blank_slice_image, bg="black")
        self.waveform_slice_image_label.grid(row=0, column=0, sticky="nsew")

        envelope_slice_frame = tk.LabelFrame(display_frame, text="Envelope slice", padx=5, pady=5)
        envelope_slice_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 5))
        envelope_slice_frame.rowconfigure(0, weight=1)
        envelope_slice_frame.columnconfigure(0, weight=1)
        envelope_slice_frame.configure(width=self.slice_image_size, height=self.slice_image_size)
        envelope_slice_frame.grid_propagate(False)
        self.envelope_slice_image_label = tk.Label(envelope_slice_frame, image=self._blank_slice_image, bg="black")
        self.envelope_slice_image_label.grid(row=0, column=0, sticky="nsew")

        voxel_slice_frame = tk.LabelFrame(display_frame, text="Voxel slice", padx=5, pady=5)
        voxel_slice_frame.grid(row=0, column=3, sticky="nsew")
        voxel_slice_frame.rowconfigure(0, weight=1)
        voxel_slice_frame.columnconfigure(0, weight=1)
        voxel_slice_frame.configure(width=self.slice_image_size, height=self.slice_image_size)
        voxel_slice_frame.grid_propagate(False)
        self.voxel_slice_image_label = tk.Label(voxel_slice_frame, image=self._blank_slice_image, bg="black")
        self.voxel_slice_image_label.grid(row=0, column=0, sticky="nsew")

        mapping_frame = tk.LabelFrame(frame, text="Audio mapping", padx=10, pady=10)
        mapping_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        mapping_frame.columnconfigure(0, weight=1)
        mapping_frame.columnconfigure(1, weight=1)
        mapping_frame.columnconfigure(2, weight=1)

        self._build_plot_canvas(frame)

        waveform_section = tk.LabelFrame(mapping_frame, text="Waveform manipulation", padx=10, pady=10)
        waveform_section.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 0))
        waveform_section.columnconfigure(1, weight=1)

        tk.Label(waveform_section, text="Waveform:").grid(row=0, column=0, sticky=tk.W)
        tk.OptionMenu(
            waveform_section,
            self.waveform_type_var,
            "sine",
            "square",
            "saw",
            "triangle",
            "Slice Mean",
            "Slice Std",
            "Edge Strength",
            "x-sum",
            "y-sum",
            "Noise Vector",
            "Texture Noise",
        ).grid(row=0, column=1, sticky=tk.W)
        self.waveform_slice_index_label = tk.Label(waveform_section, text="Slice index:")
        self.waveform_slice_index_label.grid(row=0, column=2, sticky=tk.W)
        self.waveform_slice_index_entry = tk.Entry(waveform_section, textvariable=self.waveform_slice_index_var, width=8)
        self.waveform_slice_index_entry.grid(row=0, column=3, sticky=tk.W)

        tk.Label(waveform_section, text="Envelope source:").grid(row=1, column=0, sticky=tk.W)
        tk.OptionMenu(
            waveform_section,
            self.envelope_curve_var,
            "Slice Mean",
            "Slice Std",
            "Edge Strength",
            "x-sum",
            "y-sum",
            "Noise Vector",
            "None",
        ).grid(row=1, column=1, sticky=tk.W)
        self.envelope_slice_index_label = tk.Label(waveform_section, text="Slice index:")
        self.envelope_slice_index_label.grid(row=1, column=2, sticky=tk.W)
        self.envelope_slice_index_entry = tk.Entry(waveform_section, textvariable=self.envelope_slice_index_var, width=8)
        self.envelope_slice_index_entry.grid(row=1, column=3, sticky=tk.W)

        tk.Label(waveform_section, text="Duration (s):").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(waveform_section, textvariable=self.duration_var, width=8).grid(row=2, column=1, sticky=tk.W)

        frequency_section = tk.LabelFrame(mapping_frame, text="Frequency", padx=10, pady=10)
        frequency_section.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=(0, 0))
        frequency_section.columnconfigure(1, weight=1)

        tk.Label(frequency_section, text="Frequency source:").grid(row=0, column=0, sticky=tk.W)
        tk.OptionMenu(frequency_section, self.frequency_source_var, "Global Mean", "Global p50", "Voxel Normalized").grid(row=0, column=1, sticky=tk.W)

        tk.Label(frequency_section, text="Freq min:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(frequency_section, textvariable=self.freq_min_var, width=8).grid(row=1, column=1, sticky=tk.W)
        tk.Label(frequency_section, text="Freq max:").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(frequency_section, textvariable=self.freq_max_var, width=8).grid(row=2, column=1, sticky=tk.W)

        tk.Label(frequency_section, text="Frequency modulation:").grid(row=3, column=0, sticky=tk.W)
        tk.OptionMenu(
            frequency_section,
            self.frequency_modulation_var,
            "None",
            "Slice Mean",
            "Slice Std",
            "Edge Strength",
            "x-sum",
            "y-sum",
            "Noise Vector",
        ).grid(row=3, column=1, sticky=tk.W)
        self.frequency_modulation_slice_index_label = tk.Label(frequency_section, text="Slice index:")
        self.frequency_modulation_slice_index_entry = tk.Entry(frequency_section, textvariable=self.frequency_modulation_slice_index_var, width=8)
        self.frequency_modulation_slice_index_label.grid(row=3, column=2, sticky=tk.W)
        self.frequency_modulation_slice_index_entry.grid(row=3, column=3, sticky=tk.W)

        tk.Label(frequency_section, text="Saturation:").grid(row=4, column=0, sticky=tk.W)
        tk.OptionMenu(
            frequency_section,
            self.saturation_source_var,
            "None",
            "Slice Mean",
            "Slice Std",
            "Edge Strength",
            "x-sum",
            "y-sum",
            "Noise Vector",
            "Texture Noise",
        ).grid(row=4, column=1, sticky=tk.W)
        self.saturation_slice_index_label = tk.Label(frequency_section, text="Slice index:")
        self.saturation_slice_index_entry = tk.Entry(frequency_section, textvariable=self.saturation_slice_index_var, width=8)
        self.saturation_slice_index_label.grid(row=4, column=2, sticky=tk.W)
        self.saturation_slice_index_entry.grid(row=4, column=3, sticky=tk.W)

        self.voxel_index_label = tk.Label(frequency_section, text="Voxel index:")
        self.voxel_index_entry = tk.Entry(frequency_section, textvariable=self.voxel_index_var, width=20)
        self.voxel_index_label.grid(row=5, column=0, sticky=tk.W)
        self.voxel_index_entry.grid(row=5, column=1, sticky=tk.W)

        display_section = tk.LabelFrame(mapping_frame, text="Display & Metadata", padx=10, pady=10)
        display_section.grid(row=0, column=2, sticky="nsew", padx=(0, 0), pady=(0, 0))
        display_section.columnconfigure(1, weight=1)

        tk.Label(display_section, text="Sample type:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(display_section, textvariable=self.sample_type_var, width=20).grid(row=0, column=1, sticky=tk.W)

        tk.Label(display_section, text="Sample name:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(display_section, textvariable=self.sample_name_var, width=20).grid(row=1, column=1, sticky=tk.W)

        tk.Label(display_section, text="Filter:").grid(row=2, column=0, sticky=tk.W)
        tk.OptionMenu(
            display_section,
            self.filter_type_var,
            "None",
            "Low pass",
            "High pass",
            "Band pass",
        ).grid(row=2, column=1, sticky=tk.W)
        tk.Label(display_section, text="Low cutoff:").grid(row=3, column=0, sticky=tk.W)
        tk.Entry(display_section, textvariable=self.filter_cutoff_low_var, width=10).grid(row=3, column=1, sticky=tk.W)
        tk.Label(display_section, text="High cutoff:").grid(row=4, column=0, sticky=tk.W)
        tk.Entry(display_section, textvariable=self.filter_cutoff_high_var, width=10).grid(row=4, column=1, sticky=tk.W)
        tk.Label(display_section, text="Amp distortion:").grid(row=5, column=0, sticky=tk.W)
        tk.Entry(display_section, textvariable=self.amplitude_distortion_amount_var, width=10).grid(row=5, column=1, sticky=tk.W)

        tk.Label(display_section, text="Waveform plot:").grid(row=6, column=0, sticky=tk.W)
        plot_mode_frame = tk.Frame(display_section)
        plot_mode_frame.grid(row=6, column=1, columnspan=2, sticky=tk.W)
        tk.Radiobutton(
            plot_mode_frame,
            text="Few cycles",
            variable=self.sample_plot_mode_var,
            value="few_cycles",
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(
            plot_mode_frame,
            text="Entire sample",
            variable=self.sample_plot_mode_var,
            value="entire_sample",
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            plot_mode_frame,
            text="Spectrum",
            variable=self.sample_plot_mode_var,
            value="spectrum",
        ).pack(side=tk.LEFT, padx=(10, 0))

        action_frame = tk.Frame(frame)
        action_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.preview_button = tk.Button(action_frame, text="Preview Sample", command=self._on_preview)
        self.preview_button.pack(side=tk.LEFT)
        tk.Button(action_frame, text="Save Sample", command=self._on_save_sample).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(action_frame, text="Stop", command=self._on_stop_playback).pack(side=tk.LEFT, padx=(10, 0))

        self.message_label = tk.Label(frame, text="Ready", anchor="w")
        self.message_label.grid(row=5, column=0, sticky="ew", pady=(10, 0))

    def _on_select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select input folder")
        if not folder:
            return
        folder_path = Path(folder)
        self.folder_label.config(text=str(folder_path))
        self._load_input_folder(folder_path)

    def _load_input_folder(self, folder: Path) -> None:
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Folder error", "Selected folder is not valid.")
            return

        nifti_files = list(folder.glob("*.nii.gz"))
        json_files = list(folder.glob("*.json"))
        if not nifti_files:
            messagebox.showerror("Input error", "No NIfTI file (*.nii.gz) found in the selected folder.")
            return
        if not json_files:
            messagebox.showerror("Input error", "No JSON metadata file found in the selected folder.")
            return

        nifti_path = nifti_files[0]
        json_path = json_files[0]

        try:
            self.volume, _, header = load_nifti_image(nifti_path)
            self.metadata = load_json_metadata(json_path)
        except Exception as exc:
            messagebox.showerror("Load error", f"Failed to load MRI data: {exc}")
            return

        self.global_stats = compute_global_statistics(self.volume)
        self.slice_stats = compute_slice_statistics(self.volume, axis=2)
        self.voxel_stats = compute_voxel_characteristics(self.volume, self._parse_voxel_index())
        characteristics = extract_nifti_characteristics(_, header, self.metadata)
        self._display_characteristics(characteristics)
        self.message_label.config(text="Loaded input data successfully.")

    def _display_characteristics(self, characteristics: Dict[str, Any]) -> None:
        lines = ["Input characteristics:"]
        lines.append(f"  Dimensions: {characteristics['dimensions']}")
        lines.append(f"  Orientation: {characteristics['image_orientation']}")
        lines.append(f"  Slice thickness: {_format_stat(characteristics['slice_thickness'])}")
        pixel_spacing = characteristics.get("pixel_spacing")
        if pixel_spacing is not None:
            lines.append(f"  Pixel spacing: {_format_stat(pixel_spacing[0])}, {_format_stat(pixel_spacing[1])}")
        lines.append(f"  Repetition time: {_format_stat(characteristics['repetition_time'])}")
        lines.append(f"  Echo time: {_format_stat(characteristics['echo_time'])}")
        lines.append(f"  Inversion time: {_format_stat(characteristics['inversion_time'])}")
        lines.append(f"  Flip angle: {_format_stat(characteristics['flip_angle'])}")
        lines.append("")
        lines.append("Global intensity statistics:")
        lines.append(f"  Mean: {_format_stat(self.global_stats['mean_intensity'])}")
        lines.append(f"  Std: {_format_stat(self.global_stats['std_intensity'])}")
        for label, value in self.global_stats["percentiles"].items():
            lines.append(f"  {label}: {_format_stat(value)}")
        lines.append("")
        lines.append("Slice-specific stats:")
        lines.append(f"  Slices: {self.slice_stats['slice_means'].shape[0]}")
        lines.append(f"  Avg slice mean: {_format_stat(float(np.mean(self.slice_stats['slice_means'])))}")
        lines.append(f"  Avg slice std: {_format_stat(float(np.mean(self.slice_stats['slice_stds'])))}")
        lines.append(f"  Avg edge strength: {_format_stat(float(np.mean(self.slice_stats['edge_strengths'])))}")
        lines.append("")
        lines.append("Voxel stats:")
        lines.append(f"  Index: {self.voxel_stats['voxel_index']}")
        lines.append(f"  Intensity: {_format_stat(self.voxel_stats['voxel_intensity'])}")
        lines.append(f"  Normalised: {_format_stat(self.voxel_stats['normalized_intensity'])}")

        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, "\n".join(lines))
        self.status_text.configure(state=tk.DISABLED)
        self._update_characteristic_plot()
        self._update_waveform_plot()
        self._update_mri_slice_display()

    def _build_plot_canvas(self, parent: tk.Widget) -> None:
        plot_container = tk.Frame(parent)
        plot_container.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        plot_container.columnconfigure(0, weight=1)
        plot_container.columnconfigure(1, weight=1)
        plot_container.rowconfigure(0, weight=1)

        char_frame = tk.LabelFrame(plot_container, text="Slice characteristics", padx=10, pady=10)
        char_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_title("")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.figure, master=char_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        wave_frame = tk.LabelFrame(plot_container, text="Sample waveform", padx=10, pady=10)
        wave_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.waveform_figure = Figure(figsize=(6, 3), dpi=100)
        self.waveform_figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.waveform_ax = self.waveform_figure.add_subplot(111)
        self.waveform_ax.grid(True)
        self.waveform_ax.set_title("")
        self.waveform_ax.set_xlabel("")
        self.waveform_ax.set_ylabel("")
        self.waveform_ax.set_xticks([])
        self.waveform_ax.set_yticks([])
        for spine in self.waveform_ax.spines.values():
            spine.set_visible(False)

        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_figure, master=wave_frame)
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_characteristic_plot(self) -> None:
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        if self.slice_stats is None:
            self.ax.text(0.5, 0.5, "Load MRI data to view slice characteristics.", ha="center", va="center", transform=self.ax.transAxes)
        else:
            x = np.linspace(0.0, 1.0, self.slice_stats["slice_means"].shape[0], dtype=np.float64)
            self.ax.plot(
                x,
                normalize_curve(self.slice_stats["slice_means"]),
                label="Slice Mean",
                alpha=1.0,
                linewidth=2.5,
            )
            self.ax.plot(
                x,
                normalize_curve(self.slice_stats["slice_stds"]),
                label="Slice Std",
                alpha=1.0,
                linewidth=2.5,
            )
            self.ax.plot(
                x,
                normalize_curve(self.slice_stats["edge_strengths"]),
                label="Edge Strength",
                alpha=1.0,
                linewidth=2.5,
            )

            extra_curves = []
            if self.waveform_type_var.get() in ["x-sum", "y-sum", "Noise Vector", "Texture Noise"]:
                extra_curves.append((f"Waveform {self.waveform_type_var.get()}", self._get_curve(self.waveform_type_var.get(), "waveform")))
            if self.envelope_curve_var.get() in ["x-sum", "y-sum", "Noise Vector"]:
                extra_curves.append((f"Envelope {self.envelope_curve_var.get()}", self._get_curve(self.envelope_curve_var.get(), "envelope")))
            if self.frequency_modulation_var.get() in ["x-sum", "y-sum", "Noise Vector"]:
                extra_curves.append((f"FM {self.frequency_modulation_var.get()}", self._get_curve(self.frequency_modulation_var.get(), "frequency")))
            if self.saturation_source_var.get() in ["x-sum", "y-sum", "Noise Vector", "Texture Noise"]:
                extra_curves.append((f"Saturation {self.saturation_source_var.get()}", self._get_curve(self.saturation_source_var.get(), "saturation")))

            for label, curve in extra_curves:
                if curve is not None and curve.size > 0:
                    normalized_curve = normalize_curve(curve)
                    x_extra = np.linspace(0.0, 1.0, normalized_curve.shape[0], dtype=np.float64)
                    self.ax.plot(x_extra, normalized_curve, label=label, alpha=1.0, linewidth=2.5)

            self.ax.set_xlim(0.0, 1.0)
            self.ax.set_ylim(0.0, 1.0)
            self.ax.legend(loc="upper right", fontsize="small")

        self.canvas.draw()

    def _render_slice_image(self, slice_data: np.ndarray, highlight_xy: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
        slice_data = np.asarray(slice_data, dtype=np.float64)
        if slice_data.ndim > 2:
            slice_data = np.mean(slice_data, axis=-1)
        if slice_data.size == 0:
            slice_data = np.zeros((1, 1), dtype=np.float64)
        min_val = float(np.min(slice_data))
        max_val = float(np.max(slice_data))
        if max_val <= min_val:
            normalized = np.zeros_like(slice_data, dtype=np.uint8)
        else:
            normalized = np.clip((slice_data - min_val) / (max_val - min_val), 0.0, 1.0)
            normalized = np.uint8(np.round(normalized * 255.0))
        image = Image.fromarray(normalized, mode="L").convert("RGB")
        orig_h, orig_w = slice_data.shape
        image = image.resize((self.slice_image_size, self.slice_image_size), Image.LANCZOS)
        if highlight_xy is not None:
            draw = ImageDraw.Draw(image)
            x = int(highlight_xy[0] * (image.width / orig_w))
            y = int(highlight_xy[1] * (image.height / orig_h))
            r = max(2, min(image.width, image.height) // 30)
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
        return ImageTk.PhotoImage(image)

    def _create_blank_slice_image(self, size: int = 240) -> ImageTk.PhotoImage:
        blank = Image.new("RGB", (size, size), color="black")
        return ImageTk.PhotoImage(blank)

    def _update_mri_slice_display(self) -> None:
        if self.volume is None:
            self.waveform_slice_image_label.config(image=self._blank_slice_image)
            self.envelope_slice_image_label.config(image=self._blank_slice_image)
            self.voxel_slice_image_label.config(image=self._blank_slice_image)
            self.waveform_slice_image = self._blank_slice_image
            self.envelope_slice_image = self._blank_slice_image
            self.voxel_slice_image = self._blank_slice_image
            return

        if self._uses_slice_curve(self.waveform_type_var.get()):
            slice_data = self._get_slice_data(self.waveform_slice_index_var.get())
            self.waveform_slice_image = self._render_slice_image(slice_data)
            self.waveform_slice_image_label.config(image=self.waveform_slice_image, text="")
        else:
            self.waveform_slice_image_label.config(image=self._blank_slice_image)
            self.waveform_slice_image = self._blank_slice_image

        if self._uses_slice_curve(self.envelope_curve_var.get()):
            slice_data = self._get_slice_data(self.envelope_slice_index_var.get())
            self.envelope_slice_image = self._render_slice_image(slice_data)
            self.envelope_slice_image_label.config(image=self.envelope_slice_image, text="")
        else:
            self.envelope_slice_image_label.config(image=self._blank_slice_image)
            self.envelope_slice_image = self._blank_slice_image

        if self.frequency_source_var.get() == "Voxel Normalized":
            voxel_index = self._parse_voxel_index()
            if voxel_index is None and self.volume is not None:
                voxel_index = tuple(int(s // 2) for s in self.volume.shape[:3])
            if voxel_index is not None:
                slice_data = np.take(self.volume, voxel_index[2], axis=2)
                if slice_data.ndim > 2:
                    slice_data = np.mean(slice_data, axis=-1)
                self.voxel_slice_image = self._render_slice_image(slice_data, highlight_xy=(voxel_index[0], voxel_index[1]))
                self.voxel_slice_image_label.config(image=self.voxel_slice_image, text="")
            else:
                self.voxel_slice_image_label.config(image=self._blank_slice_image)
                self.voxel_slice_image = self._blank_slice_image
        else:
            self.voxel_slice_image_label.config(image=self._blank_slice_image)
            self.voxel_slice_image = self._blank_slice_image

    def _update_waveform_plot(self) -> None:
        self.waveform_ax.clear()
        self.waveform_ax.grid(True)
        for spine in self.waveform_ax.spines.values():
            spine.set_visible(False)

        sample = self._create_sample()
        if sample is None or sample.size == 0:
            self.waveform_ax.set_xticks([])
            self.waveform_ax.set_yticks([])
            self.waveform_ax.text(
                0.5,
                0.5,
                "Create a sample by loading data and selecting settings.",
                ha="center",
                va="center",
                transform=self.waveform_ax.transAxes,
            )
        else:
            sample_rate = DEFAULT_SAMPLE_RATE
            if self.sample_plot_mode_var.get() == "spectrum":
                windowed = sample.astype(np.float64) * np.hanning(sample.shape[0])
                spectrum = np.abs(np.fft.rfft(windowed))
                frequencies = np.fft.rfftfreq(sample.shape[0], d=1.0 / sample_rate)
                spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-8))
                spectrum_db = spectrum_db - np.max(spectrum_db)
                self.waveform_ax.plot(frequencies, spectrum_db, color="tab:blue")
                self.waveform_ax.set_xlim(0.0, sample_rate / 2.0)
                self.waveform_ax.set_ylim(-80.0, 0.0)
                self.waveform_ax.set_xlabel("Frequency (Hz)")
                self.waveform_ax.set_ylabel("Level (dB)")
            else:
                frequency = self._get_frequency_scalar()
                self.waveform_ax.set_xticks([])
                self.waveform_ax.set_yticks([])
                self.waveform_ax.set_xlabel("")
                self.waveform_ax.set_ylabel("")
                if self.sample_plot_mode_var.get() == "entire_sample":
                    num_samples = sample.shape[0]
                else:
                    target_samples = int(np.ceil(5 * sample_rate / max(frequency, 1.0)))
                    num_samples = min(target_samples, sample.shape[0])
                num_samples = max(num_samples, 1)
                x = np.arange(num_samples, dtype=np.float64) / sample_rate
                self.waveform_ax.plot(x, sample[:num_samples], color="tab:blue")
                self.waveform_ax.set_xlim(0, x[-1] if x.shape[0] > 1 else max(1.0 / sample_rate, x[0] + (1.0 / sample_rate)))

        self.waveform_canvas.draw()

    def _on_plot_settings_changed(self, *args) -> None:
        self._update_voxel_index_visibility()
        self._update_sum_slice_visibility()
        self._update_characteristic_plot()
        self._update_waveform_plot()
        self._update_mri_slice_display()

    def _update_sum_slice_visibility(self) -> None:
        if self._uses_slice_curve(self.waveform_type_var.get()):
            self.waveform_slice_index_label.grid()
            self.waveform_slice_index_entry.grid()
        else:
            self.waveform_slice_index_label.grid_remove()
            self.waveform_slice_index_entry.grid_remove()

        if self._uses_slice_curve(self.envelope_curve_var.get()):
            self.envelope_slice_index_label.grid()
            self.envelope_slice_index_entry.grid()
        else:
            self.envelope_slice_index_label.grid_remove()
            self.envelope_slice_index_entry.grid_remove()

        if self._uses_slice_curve(self.frequency_modulation_var.get()):
            self.frequency_modulation_slice_index_label.grid()
            self.frequency_modulation_slice_index_entry.grid()
        else:
            self.frequency_modulation_slice_index_label.grid_remove()
            self.frequency_modulation_slice_index_entry.grid_remove()

        if self._uses_slice_curve(self.saturation_source_var.get()):
            self.saturation_slice_index_label.grid()
            self.saturation_slice_index_entry.grid()
        else:
            self.saturation_slice_index_label.grid_remove()
            self.saturation_slice_index_entry.grid_remove()

    def _parse_slice_index(self, text: str) -> Optional[int]:
        if self.volume is None:
            return None
        text = text.strip()
        if text.lower() == "center" or not text:
            return int(self.volume.shape[2] // 2)
        try:
            index = int(text)
            if index < 0:
                return 0
            return min(index, self.volume.shape[2] - 1)
        except ValueError:
            messagebox.showwarning(
                "Slice index",
                "Slice index must be an integer or 'center'. Using center slice.",
            )
            return int(self.volume.shape[2] // 2)

    def _update_voxel_index_visibility(self) -> None:
        if self.frequency_source_var.get() == "Voxel Normalized":
            self.voxel_index_label.grid()
            self.voxel_index_entry.grid()
        else:
            self.voxel_index_label.grid_remove()
            self.voxel_index_entry.grid_remove()

    def _parse_voxel_index(self) -> Optional[Tuple[int, int, int]]:
        text = self.voxel_index_var.get().strip()
        if text.lower() == "center" or not text:
            return None
        try:
            parts = [int(p) for p in text.split() if p.strip()]
            if len(parts) != 3:
                raise ValueError
            return tuple(parts)
        except ValueError:
            messagebox.showwarning("Voxel index", "Voxel index must have three integers separated by spaces. Using center voxel.")
            return None

    def _uses_slice_curve(self, name: str) -> bool:
        return name in {"x-sum", "y-sum", "Noise Vector", "Texture Noise"}

    def _get_slice_data(self, slice_text: str) -> np.ndarray:
        slice_index = self._parse_slice_index(slice_text)
        if slice_index is None:
            slice_index = int(self.volume.shape[2] // 2)
        slice_data = np.take(self.volume, slice_index, axis=2)
        if slice_data.ndim > 2:
            slice_data = np.mean(slice_data, axis=-1)
        return np.asarray(slice_data, dtype=np.float64)

    def _get_curve(self, name: str, source: Optional[str] = None) -> Optional[np.ndarray]:
        if self.slice_stats is None or self.volume is None:
            return None
        if name == "Slice Mean":
            return self.slice_stats["slice_means"]
        if name == "Slice Std":
            return self.slice_stats["slice_stds"]
        if name == "Edge Strength":
            return self.slice_stats["edge_strengths"]
        if name in ["x-sum", "y-sum", "Noise Vector", "Texture Noise"]:
            if source == "waveform":
                slice_text = self.waveform_slice_index_var.get()
            elif source == "envelope":
                slice_text = self.envelope_slice_index_var.get()
            elif source == "frequency":
                slice_text = self.frequency_modulation_slice_index_var.get()
            elif source == "saturation":
                slice_text = self.saturation_slice_index_var.get()
            else:
                slice_text = self.waveform_slice_index_var.get() if self.waveform_type_var.get() == name else self.envelope_slice_index_var.get()
            slice_data = self._get_slice_data(slice_text)
            if name == "x-sum":
                return np.sum(slice_data, axis=0).astype(np.float64)
            if name == "y-sum":
                return np.sum(slice_data, axis=1).astype(np.float64)
            if name == "Noise Vector":
                return mri_slice_to_noise_vector(slice_data)
            if name == "Texture Noise":
                return mri_slice_to_texture_signal(slice_data)
        return None

    def _get_frequency_scalar(self) -> float:
        if self.global_stats is None or self.voxel_stats is None:
            return float(self.freq_min_var.get() or DEFAULT_SAMPLE_RATE)
        source = self.frequency_source_var.get()
        if source == "Global p50":
            value = float(self.global_stats["percentiles"].get("p50", 0.0))
            min_val = float(self.global_stats["percentiles"].get("p10", 0.0))
            max_val = float(self.global_stats["percentiles"].get("p90", value))
        elif source == "Voxel Normalized":
            value = float(self.voxel_stats["normalized_intensity"])
            min_val = 0.0
            max_val = 1.0
        else:
            value = float(self.global_stats["mean_intensity"])
            min_val = float(self.global_stats["percentiles"].get("p10", value))
            max_val = float(self.global_stats["percentiles"].get("p90", value))
        return map_scalar_to_frequency(
            value,
            min_val,
            max_val,
            min_freq=float(self.freq_min_var.get() or DEFAULT_SAMPLE_RATE),
            max_freq=float(self.freq_max_var.get() or DEFAULT_SAMPLE_RATE),
        )

    def _get_duration(self) -> float:
        try:
            duration = float(self.duration_var.get())
            return max(0.2, min(duration, 10.0))
        except ValueError:
            return 2.0

    def _create_sample(self) -> Optional[np.ndarray]:
        waveform_type = self.waveform_type_var.get()
        waveform_curve = self._get_curve(self.waveform_type_var.get(), source="waveform") if self.waveform_type_var.get() in MRI_CURVE_SOURCES else None
        envelope_curve = self._get_curve(self.envelope_curve_var.get(), source="envelope")
        frequency_modulation_curve = self._get_curve(self.frequency_modulation_var.get(), source="frequency") if self.frequency_modulation_var.get() in MRI_DERIVED_WAVEFORMS else None
        saturation_curve = self._get_curve(self.saturation_source_var.get(), source="saturation") if self.saturation_source_var.get() in MRI_CURVE_SOURCES else None
        frequency = self._get_frequency_scalar()
        duration = self._get_duration()
        try:
            freq_min = float(self.freq_min_var.get() or DEFAULT_FREQ_LOW)
        except ValueError:
            freq_min = DEFAULT_FREQ_LOW
        try:
            freq_max = float(self.freq_max_var.get() or DEFAULT_FREQ_HIGH)
        except ValueError:
            freq_max = DEFAULT_FREQ_HIGH
        try:
            filter_cutoff_low = float(self.filter_cutoff_low_var.get())
        except ValueError:
            filter_cutoff_low = None
        try:
            filter_cutoff_high = float(self.filter_cutoff_high_var.get())
        except ValueError:
            filter_cutoff_high = None
        try:
            amplitude_distortion_amount = float(self.amplitude_distortion_amount_var.get())
        except ValueError:
            amplitude_distortion_amount = 0.0
        sample = synthesize_sample(
            sample_type=self.sample_type_var.get(),
            waveform_type=waveform_type,
            frequency=frequency,
            duration=duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            waveform_curve=waveform_curve,
            envelope_curve=envelope_curve,
            frequency_modulation_curve=frequency_modulation_curve,
            saturation_curve=saturation_curve,
            amplitude_distortion_amount=amplitude_distortion_amount,
            filter_type=self.filter_type_var.get(),
            filter_cutoff_low=filter_cutoff_low,
            filter_cutoff_high=filter_cutoff_high,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        return sample

    def _on_preview(self) -> None:
        sample = self._create_sample()
        if sample is None:
            return
        try:
            self._on_stop_playback()
            self.preview_handle = preview_audio(sample, sample_rate=DEFAULT_SAMPLE_RATE)
            self.preview_button.config(state=tk.DISABLED)
            duration_ms = int(self._get_duration() * 1000)
            self.preview_timer = self.root.after(duration_ms, self._preview_finished)
            self.message_label.config(text="Playing preview...")
        except ImportError as exc:
            messagebox.showerror("Playback error", str(exc))
        except Exception as exc:
            messagebox.showerror("Playback error", f"Could not play audio: {exc}")

    def _on_save_sample(self) -> None:
        sample = self._create_sample()
        if sample is None:
            return
        self.sample_output_dir.mkdir(parents=True, exist_ok=True)
        sample_name = self.sample_name_var.get().strip() or "mri_sample"
        base_name = Path(sample_name).stem
        wav_path = self.sample_output_dir / f"{base_name}.wav"
        metadata_path = self.sample_output_dir / f"{base_name}.json"
        try:
            save_wav(wav_path, sample, sample_rate=DEFAULT_SAMPLE_RATE)
            sample_metadata = {
                "sample_file": str(wav_path.name),
                "waveform_type": self.waveform_type_var.get(),
                "envelope_curve": self.envelope_curve_var.get(),
                "frequency_source": self.frequency_source_var.get(),
                "frequency_modulation": self.frequency_modulation_var.get(),
                "saturation_source": self.saturation_source_var.get(),
                "filter_type": self.filter_type_var.get(),
                "filter_cutoff_low": self.filter_cutoff_low_var.get(),
                "filter_cutoff_high": self.filter_cutoff_high_var.get(),
                "amplitude_distortion_amount": self.amplitude_distortion_amount_var.get(),
                "frequency_value": self._get_frequency_scalar(),
                "sample_type": self.sample_type_var.get(),
                "duration_seconds": self._get_duration(),
                "source_folder": str(self.folder_label.cget("text")),
                "input_metadata": self.metadata or {},
            }
            save_sample_metadata(metadata_path, sample_metadata)
            self.message_label.config(text=f"Saved sample to {wav_path.name}")
            messagebox.showinfo("Saved", f"Sample saved as {wav_path.name}")
        except Exception as exc:
            messagebox.showerror("Save error", f"Could not save sample: {exc}")

    def _on_stop_playback(self) -> None:
        if self.preview_timer is not None:
            try:
                self.root.after_cancel(self.preview_timer)
            except Exception:
                pass
            self.preview_timer = None

        if self.preview_handle is not None:
            try:
                self.preview_handle.stop()
                self.preview_handle.close()
            except Exception:
                pass
            self.preview_handle = None

        self.preview_button.config(state=tk.NORMAL)
        self.message_label.config(text="Stopped playback.")

    def _preview_finished(self) -> None:
        self.preview_timer = None
        if self.preview_handle is not None:
            try:
                self.preview_handle.stop()
                self.preview_handle.close()
            except Exception:
                pass
            self.preview_handle = None
        self.preview_button.config(state=tk.NORMAL)
        self.message_label.config(text="Playback finished.")

    def run(self) -> None:
        self.root.mainloop()
