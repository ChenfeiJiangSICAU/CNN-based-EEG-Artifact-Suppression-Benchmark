from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from brainvision_utils import get_default_vhdr_files, load_brainvision_fast, robust_scale
from label_raw_dataset import MixedArtifactDataset, build_label_raw_splits


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Missing dependency 'torch' in the active environment.") from exc
    return torch, nn, DataLoader


def build_model(nn_module):
    class ResidualBlock(nn_module.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            self.net = nn_module.Sequential(
                nn_module.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn_module.BatchNorm1d(channels),
                nn_module.GELU(),
                nn_module.Conv1d(channels, channels, kernel_size=1),
                nn_module.BatchNorm1d(channels),
            )
            self.act = nn_module.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class SingleChannelDenoiser(nn_module.Module):
        def __init__(self):
            super().__init__()
            hidden = 96
            self.encoder = nn_module.Sequential(
                nn_module.Conv1d(1, hidden, kernel_size=9, padding=4),
                nn_module.BatchNorm1d(hidden),
                nn_module.GELU(),
            )
            self.blocks = nn_module.Sequential(
                ResidualBlock(hidden, 1),
                ResidualBlock(hidden, 2),
                ResidualBlock(hidden, 4),
                ResidualBlock(hidden, 8),
                ResidualBlock(hidden, 16),
                ResidualBlock(hidden, 8),
                ResidualBlock(hidden, 4),
                ResidualBlock(hidden, 2),
                ResidualBlock(hidden, 1),
            )
            self.decoder = nn_module.Conv1d(hidden, 1, kernel_size=1)

        def forward(self, noisy):
            features = self.encoder(noisy)
            features = self.blocks(features)
            return noisy - self.decoder(features)

    return SingleChannelDenoiser()


def frequency_loss(torch_module, prediction, target):
    pred_fft = torch_module.fft.rfft(prediction, dim=-1)
    target_fft = torch_module.fft.rfft(target, dim=-1)
    return torch_module.mean(torch_module.abs(pred_fft - target_fft))


def run_epoch(model, loader, optimizer, scaler, device, torch_module, train: bool) -> Dict[str, float]:
    loss_fn = torch_module.nn.SmoothL1Loss()
    losses: List[float] = []
    if train:
        model.train()
    else:
        model.eval()

    autocast_dtype = torch_module.float16 if device.type == "cuda" else torch_module.bfloat16
    context = torch_module.enable_grad() if train else torch_module.no_grad()
    with context:
        for batch in loader:
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            with torch_module.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                prediction = model(noisy)
                loss = loss_fn(prediction, clean) + 0.1 * frequency_loss(torch_module, prediction, clean)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses))}


def denoise_signal(model, signal: np.ndarray, device, torch_module, window_len: int = 512, stride: int = 256) -> np.ndarray:
    prediction_sum = np.zeros_like(signal, dtype=np.float32)
    weight_sum = np.zeros_like(signal, dtype=np.float32)
    window = np.hanning(window_len).astype(np.float32)
    window = np.clip(window, 1e-3, None)
    model.eval()
    with torch_module.no_grad():
        for start in range(0, len(signal) - window_len + 1, stride):
            stop = start + window_len
            chunk = signal[start:stop].astype(np.float32)
            median, scale = robust_scale(chunk, axis=0)
            chunk_norm = ((chunk - median) / scale).astype(np.float32)
            pred = model(torch_module.from_numpy(chunk_norm[None, None, :]).to(device)).squeeze().cpu().numpy()
            pred = pred * scale + median
            prediction_sum[start:stop] += pred * window
            weight_sum[start:stop] += window
    uncovered = weight_sum == 0
    prediction_sum[uncovered] = signal[uncovered]
    weight_sum[uncovered] = 1.0
    return prediction_sum / weight_sum


def pick_best_segment(raw_eeg: np.ndarray, denoised_eeg: np.ndarray, sfreq: int, channel_names: List[str], view_sec: float = 5.0):
    view_samples = int(min(view_sec * sfreq, raw_eeg.shape[1]))
    best = None
    best_score = -1.0
    for ch_idx, ch_name in enumerate(channel_names):
        diff = np.abs(raw_eeg[ch_idx] - denoised_eeg[ch_idx])
        if diff.shape[0] <= view_samples:
            score = float(diff.mean())
            start = 0
        else:
            kernel = np.ones(view_samples, dtype=np.float32) / view_samples
            smoothed = np.convolve(diff, kernel, mode="valid")
            start = int(np.argmax(smoothed))
            score = float(smoothed[start])
        if score > best_score:
            best_score = score
            best = (ch_idx, ch_name, start, start + view_samples, score)
    return best


def plot_overlay(raw_signal: np.ndarray, denoised_signal: np.ndarray, sfreq: int, channel_name: str, start: int, stop: int, output_path: Path) -> None:
    time_axis = np.arange(start, stop) / sfreq
    raw_segment = raw_signal[start:stop]
    denoised_segment = denoised_signal[start:stop]
    residual = raw_segment - denoised_segment

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time_axis, raw_segment, linewidth=1.0, color="#d95f02", label="Before")
    axes[0].plot(time_axis, denoised_segment, linewidth=1.0, color="#1b9e77", label="After")
    axes[0].set_title(f"EEG Overlay Comparison - {channel_name}")
    axes[0].set_ylabel("uV")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(time_axis, residual, linewidth=0.9, color="#386cb0")
    axes[1].set_title("Removed Component (Before - After)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("uV")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pick_nth_segment(
    raw_eeg: np.ndarray,
    denoised_eeg: np.ndarray,
    sfreq: int,
    channel_names: List[str],
    rank: int = 2,
    view_sec: float = 5.0,
):
    view_samples = int(min(view_sec * sfreq, raw_eeg.shape[1]))
    candidates = []
    for ch_idx, ch_name in enumerate(channel_names):
        diff = np.abs(raw_eeg[ch_idx] - denoised_eeg[ch_idx])
        if diff.shape[0] <= view_samples:
            score = float(diff.mean())
            start = 0
        else:
            kernel = np.ones(view_samples, dtype=np.float32) / view_samples
            smoothed = np.convolve(diff, kernel, mode="valid")
            start = int(np.argmax(smoothed))
            score = float(smoothed[start])
        candidates.append((score, ch_idx, ch_name, start, start + view_samples))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[min(max(rank - 1, 0), len(candidates) - 1)]
    return selected[1], selected[2], selected[3], selected[4], selected[0]


def plot_training_curve(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train", color="#1b9e77", linewidth=1.8)
    ax.plot(epochs, val_loss, label="Validation", color="#d95f02", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EEG denoiser using Data_with_lable and Raw_data only.")
    parser.add_argument("--labeled-dir", type=Path, default=Path("Data_with_lable"))
    parser.add_argument("--raw-dir", type=Path, default=Path("Raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("label_raw_results"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch, nn, DataLoader = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cpu_count = max(1, os.cpu_count() or 1)
    torch.set_num_threads(cpu_count)
    try:
        torch.set_num_interop_threads(max(1, cpu_count // 2))
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_split, val_split, metadata = build_label_raw_splits(args.labeled_dir, args.raw_dir, seed=args.seed)
    train_ds = MixedArtifactDataset(train_split, seed=args.seed)
    val_ds = MixedArtifactDataset(val_split, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(nn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_label_raw_denoiser.pt"
    history: List[Dict[str, float]] = []
    best_val = float("inf")
    train_start = time.time()

    print(f"Using device: {device}")
    print(f"CPU threads: {cpu_count}")
    print(json.dumps(metadata, indent=2))

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, torch, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, torch, train=False)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"]})
        print(f"Epoch {epoch:02d} | train={train_metrics['loss']:.6f} | val={val_metrics['loss']:.6f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "metadata": metadata, "best_val_loss": best_val}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    vhdr_files = get_default_vhdr_files(args.raw_dir)
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr files found in {args.raw_dir}")
    loaded = load_brainvision_fast(vhdr_files[0])
    eeg = loaded["eeg"].astype(np.float32)
    channel_names = loaded["eeg_names"]
    source_fs = int(loaded["sfreq"])
    target_fs = int(metadata["target_fs"])
    if source_fs != target_fs:
        from scipy.signal import resample_poly

        eeg = resample_poly(eeg, target_fs, source_fs, axis=1).astype(np.float32)

    denoised = np.stack([denoise_signal(model, channel, device, torch) for channel in eeg], axis=0)
    best = pick_best_segment(eeg, denoised, target_fs, channel_names, view_sec=5.0)
    ch_idx, ch_name, start_idx, stop_idx, score = best

    plot_overlay(
        raw_signal=eeg[ch_idx],
        denoised_signal=denoised[ch_idx],
        sfreq=target_fs,
        channel_name=ch_name,
        start=start_idx,
        stop=stop_idx,
        output_path=args.output_dir / "eeg_denoise_overlay.png",
    )
    plot_training_curve(history, args.output_dir / "training_curve.png")
    np.save(args.output_dir / "denoised_eeg.npy", denoised.astype(np.float32))

    summary = {
        "device": str(device),
        "best_val_loss": best_val,
        "train_time_sec": time.time() - train_start,
        "best_display_channel": ch_name,
        "best_display_segment": [int(start_idx), int(stop_idx)],
        "best_display_score": float(score),
        "history": history,
        "metadata": metadata,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved overlay to {args.output_dir / 'eeg_denoise_overlay.png'}")
    print(f"Saved training curve to {args.output_dir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
