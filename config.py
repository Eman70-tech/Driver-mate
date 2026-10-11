import os
import joblib
import numpy as np 

np.random.seed(42) 

ORIG_SR    = 500    # Hz  — original Ford dataset sample rate
TARGET_SR  = 8000   # Hz  — resample target (8 kHz)
N_MFCC     = 13     # number of MFCC coefficients to extract
N_FFT      = 256    # FFT window size (samples)
HOP_LENGTH = 128    # hop length between frames (samples)

DATA_DIR   = r"D:\Driver mate\load"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


COLORS = {"healthy": "#2196F3", "faulty": "#F44336",
          "aug":     "#FF9800", "scaled": "#4CAF50"}

"""
Every parameter below is a deliberate engineering decision.
They must be IDENTICAL in training and in production.

ORIG_SR = 500
  The Ford dataset has 500 measurements. We interpret these
  as one second of audio sampled at 500 Hz.  This is the
  natural choice: 1 sample per Hz, 1 second of recording.

TARGET_SR = 8000
  Engine noise sits below 4 kHz (Nyquist: TARGET_SR / 2).
  8 kHz is the telephony standard — efficient and sufficient.
  Resampling from 500 → 8000 Hz adds temporal resolution
  that librosa needs to compute meaningful FFT windows.

N_MFCC = 13
  The classic speech/audio choice; captures the spectral
  envelope without overfitting to fine-grained frequency bins.
  Can be raised to 20-40 for more expressive features.

N_FFT = 256
  FFT window size in samples (@ 8 kHz ≈ 32 ms per window).
  Smaller than the default 2048 to match the short signal.

HOP_LENGTH = 128
  Step between consecutive FFT windows (50% overlap with 256).
  Determines the number of time frames in the MFCC output.
  n_frames = ceil(8000 / 128) = 63
"""

# Augmentation hyper-parameters (tunable)
AUG_NOISE_SNR_DB = 10    # dB — signal-to-noise ratio for white noise
AUG_STRETCH_MIN  = 0.85  # minimum time-stretch rate (slower engine)
AUG_STRETCH_MAX  = 1.15  # maximum time-stretch rate (faster engine)
AUG_AMP_LOW      = 0.70  # minimum amplitude scale factor
AUG_AMP_HIGH     = 1.30  # maximum amplitude scale factor
AUG_FREQ_MASK_F  = 3     # number of MFCC rows to zero-out per mask