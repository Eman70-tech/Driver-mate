# Driver-Mate — Ford Engine Sound Classifier

AI-powered acoustic fault detection for Ford engines. Driver-Mate converts engine audio into MFCC "fingerprints", classifies them with a Conv2D deep learning model, and serves predictions through a FastAPI backend that powers the Driver-Mate Android app (Flutter), which records engine audio on a phone in a real garage.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

---

## Table of Contents

- [Overview](#overview)
- [Mobile App](#mobile-app)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training Pipeline](#training-pipeline)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [Configuration](#configuration)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [Author](#author)

---

## Overview

| | |
|---|---|
| **Task** | Binary classification: Healthy vs. Faulty engine |
| **Datasets** | FordA and FordB (UCR Time Series Archive) — FordB's test set is recorded under **noisy conditions**, giving a real domain shift |
| **Raw input** | 500 amplitude measurements per sample (time series @ 500 Hz) |
| **Features** | MFCC matrices of shape `(13, 63)`, fed to the model as `(N, 13, 63, 1)` |
| **Model** | Conv2D neural network (Keras) |
| **Serving** | FastAPI + Uvicorn, containerised with Docker |

The key design goal: **the training path and the production path are identical**, so the model sees the same feature space on a phone microphone as it did during training.

---

## Mobile App

Driver-Mate is deployed as an Android application built with Flutter. The app records a 5-second engine sample on the phone, uploads it to this FastAPI backend, and displays the diagnosis (Healthy / Faulty) to the user.

| Component | Responsibility | Repository |
|---|---|---|
| ML model + FastAPI backend | Feature extraction, inference, majority voting | this repo |
| Flutter mobile app | Audio recording, upload, result display | [AhmedAbdelaal345/driver_mate](https://github.com/AhmedAbdelaal345/driver_mate) |

---

## How It Works

### Feature extraction

1. Cast the signal to `float32`.
2. Resample 500 Hz → **8 kHz** (Kaiser-best), removing the gap between factory sensors and phone microphones.
3. Extract **13 MFCCs** (`n_fft=256`, `hop_length=128`) → matrix of shape `(13, 63)`.

### Data augmentation (training data only)

| Augmentation | Domain | Simulates |
|---|---|---|
| White noise (SNR 20 dB) | Waveform | Background garage noise |
| Time stretch (0.85–1.15×) | Waveform | RPM variation |
| Amplitude scaling (0.70–1.30×) | Waveform | Microphone distance / sensitivity |
| Frequency masking (SpecAugment, F=3) | MFCC | Reverberation, poor mic response, regularisation |

This grows the training set ~5× (≈3.6k → ≈18k samples). Test sets are never augmented.

> The `StandardScaler` is fitted on the **original, un-augmented** training data only, then applied to all splits. This avoids learning the statistics of synthetic noise.

### Live inference with majority voting

A mobile recording (default 5 s) is split into 1-second chunks. Each chunk becomes a `(13, 63, 1)` MFCC tensor and is scored independently:

```text
recording (5 s @ 8 kHz)
   └─▶ 5 × 1-second chunks
         └─▶ MFCC → scale → Conv2D model → P(Healthy)
               └─▶ P(Healthy) ≥ 0.847  →  "Healthy" vote
                     └─▶ engine is Healthy only if
                         healthy_votes ≥ ceil(n_chunks × 0.80)
```

The decision threshold (`0.847`) is deliberately conservative: it favours catching faults over missing them. Majority voting makes the system robust to transient noise (a horn, a door slam) in any single chunk.

---

## Project Structure

```text
Driver-mate/
├── api.py                 # FastAPI app (/health, /predict)
├── inference.py           # Production inference + majority voting
├── audio_processing.py    # MFCC extraction (training + live audio)
├── augmentation.py        # Audio & MFCC augmentations
├── scaling.py             # StandardScaler fit/apply, label encoding, reshape
├── data_loader.py         # Raw FordA/FordB loading and class-balance checks
├── model_builder.py       # Conv2D model definition
├── train.py               # Model training
├── evaluate.py            # Evaluation and threshold tuning
├── visualize_mfcc.py      # Waveform / MFCC / class-difference plots
├── config.py              # Single source of truth for all parameters
├── main_prep.py           # Pipeline stage 1: MFCC extraction
├── main_part2.py          # Pipeline stage 2: augmentation
├── main_finish.py         # Pipeline stage 3: scaling & final tensors
├── run_pipeline.py        # Runs stages 1–3 sequentially
├── artifacts/             # Model, scaler, optimal threshold
├── HANDOVER_GUIDE.md      # Backend integration guide
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `libsndfile` (required by `soundfile`; on Debian/Ubuntu: `sudo apt install libsndfile1`)

### Installation

```bash
git clone https://github.com/Eman70-tech/Driver-mate.git
cd Driver-mate

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Data

Download the **FordA** and **FordB** datasets from the [UCR Time Series Classification Archive](https://www.timeseriesclassification.com/) and place the `.txt` files in the directory referenced by `DATA_DIR` in `config.py`.

> Raw data and generated `.npy` files are not stored in this repository. FordA and FordB remain subject to the terms of the UCR archive.

---

## Training Pipeline

Run all three stages in one command:

```bash
python run_pipeline.py
```

| Stage | Script | What it does | Output |
|---|---|---|---|
| 1 | `main_prep.py` | Load raw data → extract MFCCs for all four splits | `mfcc_*.npy` |
| 2 | `main_part2.py` | Apply 4 augmentations to training sets, concatenate (5×), shuffle | `mfcc_*_aug.npy` |
| 3 | `main_finish.py` | Fit scalers, scale, encode labels, reshape to `(N,13,63,1)` | `final_*.npy`, `scaler_*.pkl` |

Then train and evaluate:

```bash
python train.py
python evaluate.py
```

---

## Running the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use `--workers 1`: TensorFlow is not fork-safe, and the model is loaded once at startup.

**Health check**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Predict**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@engine_recording.wav"
```

The response contains the final prediction, a confidence score, and the per-chunk votes (one entry per 1-second chunk). Interactive docs are available at `http://localhost:8000/docs`.

**Recording guidelines**

- WAV format, ideally **5 seconds** long.
- Shorter recordings are zero-padded; longer ones are cropped.

For backend integration details, see the [Handover Guide](HANDOVER_GUIDE.md).

---

## Docker

```bash
docker build -t driver-mate .
docker run -p 8000:8000 driver-mate
```

Make sure the `artifacts/` folder (model, scaler, threshold) is available to the container.

---

## Configuration

All parameters live in `config.py`. The following **must be identical between training and production**:

| Parameter | Value |
|---|---|
| `ORIG_SR` | 500 Hz |
| `TARGET_SR` | 8000 Hz |
| `N_MFCC` | 13 |
| `N_FFT` | 256 |
| `HOP_LENGTH` | 128 |

Changing any one of these without retraining will cause silent inference errors.

---

## Known Limitations & Roadmap

- [ ] Add CORS middleware for Flutter web clients
- [ ] Add rate limiting on `/predict`
- [ ] Add API-key authentication before internet exposure
- [ ] Return a `severity` field based on vote distribution
- [ ] Silent-audio guard before inference
- [ ] Validate `Content-Type` in addition to file extension
- [ ] Log the model version (hash) at startup
- [ ] Add unit tests and CI (GitHub Actions)

The model is trained on sensor data resampled to 8 kHz, not on raw microphone recordings, so real-world performance on phone audio should be validated before relying on it.

---

## Author

**Eman** — Computer Science graduate, Zagazig University · Machine Learning & AI Engineering

[LinkedIn](https://www.linkedin.com/in/eman-ahmed-ai) · [GitHub](https://github.com/Eman70-tech)