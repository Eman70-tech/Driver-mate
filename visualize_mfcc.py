import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import librosa
import librosa.display

import config
from data_loader import load_dataset
from audio_processing import process_single_audio

def plot_single_sample():
    print("📊 Visualising the MFCC pipeline on a single sample...")
    
    X_trainA, y_trainA, _, _ = load_dataset(config.DATA_DIR, "FordA")
    
    idx_healthy = int(np.where(y_trainA ==  1)[0][0])
    idx_faulty  = int(np.where(y_trainA == -1)[0][0])

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("MFCC Extraction Pipeline — Single Sample Visualisation\n"
                 "(FordA Train: one Healthy, one Faulty)",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    for col, (idx, label, class_name, colour) in enumerate([
        (idx_healthy,  1, "Healthy (+1)", config.COLORS["healthy"]),
        (idx_faulty,  -1, "Faulty  (-1)", config.COLORS["faulty"]),
    ]):
        raw_signal = X_trainA[idx]  # shape (500,)

        mfccs = process_single_audio(raw_signal)

        ax_mfcc = fig.add_subplot(gs[2, col])
        img = librosa.display.specshow(
            mfccs,
            sr         = config.TARGET_SR,
            hop_length = config.HOP_LENGTH,
            x_axis     = "time",
            ax         = ax_mfcc,
            cmap       = "coolwarm"
        )
        fig.colorbar(img, ax=ax_mfcc, format="%+.1f")
        ax_mfcc.set_title(f"[{class_name}] MFCC — shape {mfccs.shape}", fontsize=10, fontweight="bold")
        ax_mfcc.set_xlabel("Time (s)")
        ax_mfcc.set_ylabel("MFCC Coefficient")

    save_path = f"{config.OUTPUT_DIR}/P1_02_mfcc_pipeline_single_sample.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Figure saved → {save_path}\n")


def plot_class_differences():
    """
    Load saved MFCC arrays and plot the difference between Healthy and Faulty engines.
    """
    print("📊 Loading saved MFCCs for difference visualization...")
    
    try:
        mfcc_trainA = np.load(f"{config.OUTPUT_DIR}/mfcc_trainA.npy")
        y_trainA    = np.load(f"{config.OUTPUT_DIR}/y_trainA_raw.npy")
        mfcc_testA  = np.load(f"{config.OUTPUT_DIR}/mfcc_testA.npy")
        y_testA     = np.load(f"{config.OUTPUT_DIR}/y_testA_raw.npy")
        
        mfcc_trainB = np.load(f"{config.OUTPUT_DIR}/mfcc_trainB.npy")
        y_trainB    = np.load(f"{config.OUTPUT_DIR}/y_trainB_raw.npy")
        mfcc_testB  = np.load(f"{config.OUTPUT_DIR}/mfcc_testB.npy")
        y_testB     = np.load(f"{config.OUTPUT_DIR}/y_testB_raw.npy")
    except FileNotFoundError:
        print(" Error: .npy files not found! Please run main_prep.py first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Mean MFCC Spectrogram per Class\n(Divergence = discriminative region)", 
                 fontsize=13, fontweight="bold")

    plot_configs = [
        (mfcc_trainA, y_trainA, "FordA TRAIN", axes[0, 0]),
        (mfcc_testA,  y_testA,  "FordA TEST",  axes[0, 1]),
        (mfcc_trainB, y_trainB, "FordB TRAIN", axes[1, 0]),
        (mfcc_testB,  y_testB,  "FordB TEST",  axes[1, 1]),
    ]

    for mfcc_data, y_raw, name, ax in plot_configs:
        mean_healthy = mfcc_data[y_raw == 1].mean(axis=0)  
        mean_faulty  = mfcc_data[y_raw == -1].mean(axis=0) 
        
        diff = mean_healthy - mean_faulty

        im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Healthy − Faulty")
        ax.set_title(f"{name} — MFCC Difference", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("MFCC Coefficient Index")
        ax.set_yticks(range(config.N_MFCC))

    save_path = os.path.join(config.OUTPUT_DIR, "mfcc_class_difference.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" Figure saved successfully to: {save_path}\n")

# Main Execution Box
if __name__ == "__main__":
    plot_single_sample()
    plot_class_differences()