from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_EOG_NAMES = {"HEO", "VEO", "HEOG", "VEOG", "EOG"}


def get_default_vhdr_files(raw_dir: Path) -> List[Path]:
    return sorted(raw_dir.glob("*.vhdr"))


def parse_brainvision_header(vhdr_path: Path) -> Dict[str, object]:
    raw_text = vhdr_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("["):
            start_idx = idx
            break
    text = "\n".join(lines[start_idx:])
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    cfg.optionxform = str
    cfg.comment_prefixes = (";",)
    cfg.read_string(text)

    common = cfg["Common Infos"]
    binary = cfg["Binary Infos"]
    channels = cfg["Channel Infos"]

    data_file = vhdr_path.parent / common["DataFile"]
    n_channels = int(common["NumberOfChannels"])
    sampling_interval_us = float(common["SamplingInterval"])
    sfreq = 1e6 / sampling_interval_us
    binary_format = binary["BinaryFormat"].strip().upper()
    if binary_format != "IEEE_FLOAT_32":
        raise ValueError(f"Unsupported BrainVision binary format: {binary_format}")

    ch_names: List[str] = []
    for idx in range(1, n_channels + 1):
        raw_entry = channels[f"Ch{idx}"]
        name = raw_entry.split(",")[0].strip()
        ch_names.append(name)

    return {
        "data_file": data_file,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "ch_names": ch_names,
    }


def load_brainvision_fast(vhdr_path: Path) -> Dict[str, object]:
    header = parse_brainvision_header(vhdr_path)
    data_file = Path(header["data_file"])
    n_channels = int(header["n_channels"])
    sfreq = float(header["sfreq"])
    ch_names = list(header["ch_names"])

    raw = np.fromfile(data_file, dtype="<f4")
    if raw.size % n_channels != 0:
        raise ValueError(f"Data size in {data_file.name} is not divisible by channel count.")

    signal = raw.reshape(-1, n_channels).T.astype(np.float32)
    eog_indices = [idx for idx, name in enumerate(ch_names) if name.upper() in DEFAULT_EOG_NAMES]
    eeg_indices = [idx for idx in range(len(ch_names)) if idx not in eog_indices]

    eeg = signal[eeg_indices]
    eog = signal[eog_indices] if eog_indices else np.zeros((0, signal.shape[1]), dtype=np.float32)
    eeg_names = [ch_names[idx] for idx in eeg_indices]
    eog_names = [ch_names[idx] for idx in eog_indices]

    return {
        "signal": signal,
        "eeg": eeg,
        "eog": eog,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "eeg_names": eeg_names,
        "eog_names": eog_names,
    }


def channel_artifact_weights(channel_names: List[str]) -> np.ndarray:
    weights = []
    for name in channel_names:
        upper = name.upper()
        if upper.startswith(("FP", "AF")):
            weights.append(1.0)
        elif upper.startswith(("F", "FC")):
            weights.append(0.8)
        elif upper.startswith(("C", "T")):
            weights.append(0.45)
        elif upper.startswith(("CP", "TP", "P")):
            weights.append(0.2)
        elif upper.startswith(("PO", "O")):
            weights.append(0.1)
        else:
            weights.append(0.35)
    return np.asarray(weights, dtype=np.float32)


def robust_scale(signal: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    median = np.median(signal, axis=axis, keepdims=True)
    mad = np.median(np.abs(signal - median), axis=axis, keepdims=True)
    scale = np.clip(1.4826 * mad, 1e-6, None)
    return median.astype(np.float32), scale.astype(np.float32)


def slice_windows(
    eeg: np.ndarray,
    eog: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    total_samples = eeg.shape[1]
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, total_samples - window_samples + 1, stride_samples):
        stop = start + window_samples
        windows.append((eeg[:, start:stop], eog[:, start:stop]))
    return windows


def save_metadata(output_dir: Path, metadata: Dict[str, object]) -> None:
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
