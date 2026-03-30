from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from brainvision_utils import get_default_vhdr_files, load_brainvision_fast


@dataclass
class SplitData:
    clean_windows: List[np.ndarray]
    eog_artifacts: List[np.ndarray]
    emg_artifacts: List[np.ndarray]


class MixedArtifactDataset:
    def __init__(
        self,
        split_data: SplitData,
        seed: int = 42,
        clean_probability: float = 0.15,
    ):
        self.clean_windows = split_data.clean_windows
        self.eog_artifacts = split_data.eog_artifacts
        self.emg_artifacts = split_data.emg_artifacts
        self.clean_probability = clean_probability
        self.rng = np.random.default_rng(seed)
        if not self.clean_windows:
            raise ValueError("No clean windows provided.")

    def __len__(self) -> int:
        return len(self.clean_windows)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        clean = self.clean_windows[index].astype(np.float32)
        median, scale = _robust_scale(clean)
        clean_norm = (clean - median) / scale

        noisy = clean.copy()
        artifact_name = "clean"
        if self.rng.random() > self.clean_probability:
            artifact_name = "eog" if self.rng.random() < 0.5 else "emg"
            artifact_pool = self.eog_artifacts if artifact_name == "eog" else self.emg_artifacts
            artifact = artifact_pool[int(self.rng.integers(0, len(artifact_pool)))].astype(np.float32)
            artifact = artifact - np.median(artifact)
            artifact /= np.std(artifact) + 1e-6
            amplitude = self.rng.uniform(0.25, 1.15) * (np.std(clean) + 1e-6)
            noisy = clean + amplitude * artifact

        noisy_norm = (noisy - median) / scale
        return {
            "noisy": noisy_norm[None, :].astype(np.float32),
            "clean": clean_norm[None, :].astype(np.float32),
            "raw_noisy": noisy.astype(np.float32),
            "raw_clean": clean.astype(np.float32),
            "artifact_name": artifact_name,
        }


def build_label_raw_splits(
    labeled_dir: Path,
    raw_dir: Path,
    seed: int = 42,
    target_fs: int = 256,
    window_len: int = 512,
    stride: int = 256,
    val_ratio: float = 0.2,
    max_raw_windows: int = 12000,
) -> Tuple[SplitData, SplitData, Dict[str, object]]:
    rng = np.random.default_rng(seed)

    eeg_epochs = _load_npy_or_mat(labeled_dir, "EEG_all_epochs").astype(np.float32)
    eog_epochs = _load_npy_or_mat(labeled_dir, "EOG_all_epochs").astype(np.float32)
    emg_epochs = _load_npy_or_mat(labeled_dir, "EMG_all_epochs").astype(np.float32)

    raw_windows = _extract_raw_windows(raw_dir, target_fs=target_fs, window_len=window_len, stride=stride)
    if len(raw_windows) > max_raw_windows:
        keep_indices = rng.choice(len(raw_windows), size=max_raw_windows, replace=False)
        raw_windows = [raw_windows[idx] for idx in keep_indices]

    clean_windows = [epoch.astype(np.float32) for epoch in eeg_epochs] + raw_windows
    rng.shuffle(clean_windows)

    train_clean, val_clean = _split_sequence(clean_windows, val_ratio)
    train_eog, val_eog = _split_array(eog_epochs, val_ratio, rng)
    train_emg, val_emg = _split_array(emg_epochs, val_ratio, rng)

    train_split = SplitData(
        clean_windows=[x.astype(np.float32) for x in train_clean],
        eog_artifacts=[x.astype(np.float32) for x in train_eog],
        emg_artifacts=[x.astype(np.float32) for x in train_emg],
    )
    val_split = SplitData(
        clean_windows=[x.astype(np.float32) for x in val_clean],
        eog_artifacts=[x.astype(np.float32) for x in val_eog],
        emg_artifacts=[x.astype(np.float32) for x in val_emg],
    )

    metadata = {
        "target_fs": target_fs,
        "window_len": window_len,
        "stride": stride,
        "clean_epoch_count": int(len(eeg_epochs)),
        "raw_window_count": int(len(raw_windows)),
        "train_count": int(len(train_split.clean_windows)),
        "val_count": int(len(val_split.clean_windows)),
        "eog_count": int(len(eog_epochs)),
        "emg_count": int(len(emg_epochs)),
    }
    return train_split, val_split, metadata


def _load_npy_or_mat(directory: Path, stem: str) -> np.ndarray:
    npy_path = directory / f"{stem}.npy"
    if npy_path.exists():
        return np.load(npy_path)
    mat_path = directory / f"{stem}.mat"
    if mat_path.exists():
        data = loadmat(mat_path)
        return data[stem]
    raise FileNotFoundError(f"Missing {stem}.npy or {stem}.mat in {directory}")


def _extract_raw_windows(raw_dir: Path, target_fs: int, window_len: int, stride: int) -> List[np.ndarray]:
    windows: List[np.ndarray] = []
    for vhdr_path in get_default_vhdr_files(raw_dir):
        loaded = load_brainvision_fast(vhdr_path)
        eeg = loaded["eeg"].astype(np.float32)
        source_fs = int(loaded["sfreq"])
        if source_fs != target_fs:
            eeg = resample_poly(eeg, target_fs, source_fs, axis=1).astype(np.float32)
        for channel in eeg:
            centered = channel - np.median(channel)
            for start in range(0, centered.shape[0] - window_len + 1, stride):
                windows.append(centered[start : start + window_len].astype(np.float32))
    return windows


def _split_sequence(sequence: Sequence[np.ndarray], val_ratio: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n_val = max(1, int(len(sequence) * val_ratio))
    return list(sequence[n_val:]), list(sequence[:n_val])


def _split_array(array: np.ndarray, val_ratio: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(len(array))
    n_val = max(1, int(len(array) * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return array[train_idx], array[val_idx]


def _robust_scale(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    median = np.median(signal, keepdims=True)
    mad = np.median(np.abs(signal - median), keepdims=True)
    scale = np.clip(1.4826 * mad, 1e-6, None)
    return median.astype(np.float32), scale.astype(np.float32)
