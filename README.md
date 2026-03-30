# EEG Artifact Suppression Benchmark

## Description

**EEG Artifact Suppression Benchmark** is a conservative single-channel CNN project for EEG denoising under limited supervision. It combines labeled EEG/EOG/EMG epochs from `Data_with_lable` with real multi-channel BrainVision recordings from `Raw_data` to build a pseudo-supervised training pipeline for artifact suppression.

Rather than pursuing visually aggressive waveform modification, this project focuses on stable and interpretable suppression of ocular and muscle artifacts while preserving the original EEG morphology as much as possible. It is intended as a reproducible baseline for EEG denoising when true paired `noisy-clean` targets are unavailable.

## Project Objective

- Build a complete EEG denoising pipeline from BrainVision raw data loading to model inference
- Use labeled artifact data to construct supervised training samples
- Train a robust single-channel baseline under the absence of true paired clean targets
- Provide a reference implementation for future multi-channel and artifact-aware EEG denoising models

## Data Sources and Task Definition

### 1. Raw_data

- Format: BrainVision `vhdr/eeg/vmrk`
- Original sampling rate: `1000 Hz`
- The pipeline parses header metadata and loads binary EEG signals directly
- EEG and EOG channels are automatically separated based on channel names

This dataset is the closest to the real application scenario, but it does not provide true paired `clean/noisy` supervision.

### 2. Data_with_lable

- `EEG_all_epochs`: relatively clean EEG segments
- `EOG_all_epochs`: ocular artifact segments
- `EMG_all_epochs`: muscle artifact segments

The project treats this dataset as a combination of a clean signal pool and an artifact pool for synthetic supervised sample generation.

### 3. Task Definition

The project follows a pseudo-supervised formulation:

```text
noisy = clean + artifact
target = clean
```

Therefore, the model is trained for artifact suppression rather than strict reconstruction of the true clean waveform.

## Overall Technical Pipeline

1. Parse BrainVision EEG recordings and extract real single-channel EEG windows
2. Load labeled EEG, EOG, and EMG epochs to build clean and artifact repositories
3. Randomly inject EOG or EMG artifacts into clean EEG windows during training
4. Apply robust scaling to each window to reduce the influence of outliers
5. Train a single-channel residual 1D CNN to learn the mapping from `noisy` to `clean`
6. Perform sliding-window denoising on real EEG recordings and reconstruct full signals with overlap-add
7. Export denoised signals, training curves, and waveform overlays for qualitative analysis

## Methods and Technical Details by Component

### Data Loading and Parsing

Reference file: `brainvision_utils.py`

Techniques:

- BrainVision header parsing with `configparser`
- Direct binary signal loading with `numpy.fromfile`
- Automatic identification of EOG channels such as `HEO`, `VEO`, `HEOG`, `VEOG`, and `EOG`
- Conversion of signals into `channels x samples` layout for downstream slicing and inference

Technical rationale:

- Fast signal loading
- Minimal dependency on heavy EEG toolchains
- Direct support for real BrainVision recordings in an end-to-end workflow

### Data Preprocessing

Reference files: `label_raw_dataset.py`, `brainvision_utils.py`

Techniques:

- Resample `Raw_data` to `256 Hz`
- Segment signals with a sliding window of length `512` and stride `256`
- Apply median-centering to each channel before slicing
- Normalize each segment with robust scaling based on `median + MAD`
- Randomly subsample raw windows to at most `12000` samples to control training size

Technical rationale:

- Align training and inference input distributions through unified sampling rate
- Increase sample count and support local temporal modeling through windowing
- Improve robustness to large-amplitude outliers common in EEG recordings
- Reduce the impact of baseline drift and abnormal amplitude spikes

### Training Sample Construction

Reference file: `label_raw_dataset.py`

Techniques:

- Use `EEG_all_epochs` and windows extracted from `Raw_data` as the clean pool
- Use `EOG_all_epochs` and `EMG_all_epochs` as two artifact repositories
- Keep some samples clean and synthesize others by injecting EOG or EMG artifacts
- Normalize artifacts before mixing and scale them with random amplitudes
- Construct training pairs online instead of precomputing a fixed dataset

Technical rationale:

- Combine labeled artifact data with real EEG distribution
- Expose the model to diverse artifact types and amplitudes
- Reduce overfitting to fixed training pairs
- Enable supervised learning despite the lack of real paired denoising labels

### Model Architecture

Reference file: `label_raw_train.py`

Techniques:

- Single-channel `1D CNN` architecture
- Initial temporal encoder with `Conv1d(1 -> 96, kernel_size=9)`
- Backbone composed of 9 residual blocks with dilation schedule `1, 2, 4, 8, 16, 8, 4, 2, 1`
- Each residual block uses `Conv1d + BatchNorm1d + GELU + 1x1 Conv1d + BatchNorm1d`
- Final output predicts a residual component and produces the denoised signal via `noisy - residual`

Technical rationale:

- 1D CNNs are well suited for temporal EEG modeling
- Dilated convolutions enlarge the receptive field without excessive depth
- Residual prediction encourages selective noise removal instead of full waveform rewriting
- Single-channel design keeps the baseline simple and computationally efficient

### Loss Function and Training Strategy

Reference file: `label_raw_train.py`

Techniques:

- Primary loss: `SmoothL1Loss`
- Auxiliary frequency-domain loss based on `rFFT` discrepancy
- Total objective: `SmoothL1 + 0.1 * FrequencyLoss`
- Optimizer: `AdamW`
- Gradient clipping with `max_norm=5.0`
- Automatic mixed precision on CUDA when available

Technical rationale:

- `SmoothL1Loss` is more stable than pure MSE under outliers
- Frequency-domain supervision helps preserve spectral structure
- `AdamW` improves optimization stability
- Gradient clipping reduces the risk of unstable updates

### Inference and Signal Reconstruction

Reference file: `label_raw_train.py`

Techniques:

- Apply denoising to each EEG channel independently
- Normalize each inference window with the same robust scaling strategy
- Restore original scale after prediction
- Reconstruct the full signal with `Hanning window + overlap-add`
- Preserve uncovered boundary samples from the original signal

Technical rationale:

- Keeps train-time and test-time normalization behavior consistent
- Reduces block boundary artifacts during reconstruction
- Improves stability for full-length EEG restoration

### Result Analysis and Visualization

Reference files: `label_raw_train.py`, `label_raw_results/analysis.json`

Techniques:

- Save train/validation loss curves
- Automatically select the segment with the most visible denoising difference
- Plot raw signal, denoised signal, and removed residual component
- Export denoised EEG arrays and experiment metadata

Current experiment summary:

- Target sampling rate: `256 Hz`
- Window length: `512`
- Stride: `256`
- Clean EEG epoch count: `4514`
- Raw window count: `12000`
- Training sample count: `13212`
- Validation sample count: `3302`
- Best validation loss: `0.7865`
- Training device: `cuda`

## Strengths

- Explicitly uses labeled EOG and EMG artifact data, making the supervision more interpretable
- Incorporates real raw EEG windows into the clean pool, improving practical data coverage
- Uses robust normalization that is more stable under abnormal amplitudes
- Residual single-channel CNN behaves conservatively and is less likely to overwrite the original waveform aggressively
- Includes a complete engineering pipeline from raw BrainVision input to denoised output
- Serves as a reproducible baseline under the current data constraints

## Limitations

- No true paired `noisy-clean` supervision is available; the training target is still synthetic
- The current model is single-channel and does not exploit spatial correlations across EEG channels
- The artifact model mainly covers EOG and EMG contamination and cannot fully represent all real motion-related artifacts
- The model is biased toward conservative suppression, so the visible denoising effect may appear limited
- The supervision setup encourages additive artifact removal more than reliable reconstruction of a physiologically valid clean waveform
- Evaluation is currently centered on loss values and qualitative plots, with limited comparison against classical EEG denoising baselines such as ICA or ASR

## Intended Positioning

This project is better viewed as:

- a single-channel EEG artifact suppression baseline under constrained supervision
- a runnable BrainVision EEG preprocessing and denoising prototype
- a precursor to future multi-channel artifact-aware denoising systems

It should not be interpreted as a final high-confidence clean EEG reconstruction model.

## Future Directions

- Introduce multi-channel labeled EEG datasets with richer artifact annotations
- Upgrade the model to a multi-channel spatiotemporal architecture
- Add artifact-type conditioning for conditional denoising
- Expand evaluation with spectral fidelity, cross-subject generalization, and spatial consistency metrics
- Benchmark against classical EEG denoising approaches such as ICA and ASR
