import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import config
from data_loader import load_dataset
from augmentation import (augment_white_noise, augment_time_stretch, 
                          augment_amplitude, augment_freq_mask, 
                          augment_batch_audio, augment_batch_mfcc)

def plot_augmentation_sanity_check(sample_raw, title_prefix="FordA"):
    print(f"\nPlotting augmentation effects on one {title_prefix} sample...")
    
    audio_8k = librosa.resample(sample_raw.astype(np.float32), 
                                orig_sr=config.ORIG_SR, target_sr=config.TARGET_SR)

    aug_examples = {
        "Original":              librosa.feature.mfcc(y=audio_8k, sr=config.TARGET_SR, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH),
        "① White Noise":         librosa.feature.mfcc(y=augment_white_noise(audio_8k), sr=config.TARGET_SR, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH),
        "② Time Stretch (×0.9)": librosa.feature.mfcc(y=augment_time_stretch(audio_8k, rate=0.90), sr=config.TARGET_SR, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH),
        "③ Amplitude (×0.7)":    librosa.feature.mfcc(y=augment_amplitude(audio_8k, 0.70, 0.70), sr=config.TARGET_SR, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH),
        "④ Freq Mask":           augment_freq_mask(librosa.feature.mfcc(y=audio_8k, sr=config.TARGET_SR, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    fig.suptitle(f"Augmentation Effects — MFCC of One {title_prefix} Sample", fontsize=13, fontweight="bold")

    vmin = min(m.min() for m in aug_examples.values())
    vmax = max(m.max() for m in aug_examples.values())

    for ax, (title, mfcc_eg) in zip(axes, aug_examples.items()):
        img = librosa.display.specshow(mfcc_eg, sr=config.TARGET_SR, hop_length=config.HOP_LENGTH, x_axis="time", ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax is axes[0]: ax.set_ylabel("MFCC Index")

    fig.colorbar(img, ax=axes[-1], label="MFCC Magnitude")
    plt.savefig(f"{config.OUTPUT_DIR}/P2_01_augmentation_effects_{title_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → P2_01_augmentation_effects_{title_prefix}.png")

def main():

    print(" LOADING PART 1 OUTPUTS & RAW DATA")
    

    mfcc_trainA = np.load(f"{config.OUTPUT_DIR}/mfcc_trainA.npy")
    mfcc_trainB = np.load(f"{config.OUTPUT_DIR}/mfcc_trainB.npy")

    X_trainA_raw, y_trainA_raw, _, _ = load_dataset(config.DATA_DIR, "FordA")
    X_trainB_raw, y_trainB_raw, _, _ = load_dataset(config.DATA_DIR, "FordB")

    print(" APPLYING AUGMENTATIONS (TRAIN SETS ONLY)")
 

    # ── FordA augmentation ──
    print("\n[FordA] Training Set Augmentation:")
    mfcc_trainA_noise   = augment_batch_audio(X_trainA_raw, lambda a: augment_white_noise(a), "① White Noise")
    mfcc_trainA_stretch = augment_batch_audio(X_trainA_raw, lambda a: augment_time_stretch(a), "② Time Stretch")
    mfcc_trainA_amp     = augment_batch_audio(X_trainA_raw, lambda a: augment_amplitude(a), "③ Amplitude Scale")
    mfcc_trainA_fmask   = augment_batch_mfcc(mfcc_trainA, lambda m: augment_freq_mask(m), "④ Freq Mask")

    # ── FordB augmentation ──
    print("\n[FordB] Training Set Augmentation:")
    mfcc_trainB_noise   = augment_batch_audio(X_trainB_raw, lambda a: augment_white_noise(a), "① White Noise")
    mfcc_trainB_stretch = augment_batch_audio(X_trainB_raw, lambda a: augment_time_stretch(a), "② Time Stretch")
    mfcc_trainB_amp     = augment_batch_audio(X_trainB_raw, lambda a: augment_amplitude(a), "③ Amplitude Scale")
    mfcc_trainB_fmask   = augment_batch_mfcc(mfcc_trainB, lambda m: augment_freq_mask(m), "④ Freq Mask")


    mfcc_trainA_full = np.concatenate([mfcc_trainA, mfcc_trainA_noise, mfcc_trainA_stretch, mfcc_trainA_amp, mfcc_trainA_fmask], axis=0)
    y_trainA_full = np.tile(y_trainA_raw, 5) # تكرار الـ Labels 5 مرات

    mfcc_trainB_full = np.concatenate([mfcc_trainB, mfcc_trainB_noise, mfcc_trainB_stretch, mfcc_trainB_amp, mfcc_trainB_fmask], axis=0)
    y_trainB_full = np.tile(y_trainB_raw, 5)

    print(f"\nAugmented training sets combined! Growth factor: 5x")
    print(f"   FordA TRAIN: {mfcc_trainA.shape[0]} → {mfcc_trainA_full.shape[0]} samples")
    print(f"   FordB TRAIN: {mfcc_trainB.shape[0]} → {mfcc_trainB_full.shape[0]} samples")

    rng = np.random.default_rng(42)
    idx_A = rng.permutation(len(mfcc_trainA_full))
    mfcc_trainA_full = mfcc_trainA_full[idx_A]
    y_trainA_full    = y_trainA_full[idx_A]

    idx_B = rng.permutation(len(mfcc_trainB_full))
    mfcc_trainB_full = mfcc_trainB_full[idx_B]
    y_trainB_full    = y_trainB_full[idx_B]
    print("Augmented training arrays shuffled.")


    plot_augmentation_sanity_check(X_trainA_raw[0], "FordA")
    

    np.save(f"{config.OUTPUT_DIR}/mfcc_trainA_aug.npy", mfcc_trainA_full)
    np.save(f"{config.OUTPUT_DIR}/y_trainA_aug.npy", y_trainA_full)
    np.save(f"{config.OUTPUT_DIR}/mfcc_trainB_aug.npy", mfcc_trainB_full)
    np.save(f"{config.OUTPUT_DIR}/y_trainB_aug.npy", y_trainB_full)
    print("Augmented Arrays Saved to outputs folder.")

if __name__ == "__main__":
    main()