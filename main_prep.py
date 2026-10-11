import os
import numpy as np
import config
from data_loader import load_dataset
from audio_processing import extract_features_batch

def main():
    print("Starting Data Preprocessing Pipeline (Phase 1)...")

    print("\n1. Loading Datasets")
    X_trainA, y_trainA, X_testA, y_testA = load_dataset(config.DATA_DIR, "FordA")
    X_trainB, y_trainB, X_testB, y_testB = load_dataset(config.DATA_DIR, "FordB")

    # 2. Extract MFCCs (The Heavy Lifting)
    print("\n2. Extracting MFCC Features ")
    mfcc_trainA = extract_features_batch(X_trainA)
    mfcc_testA  = extract_features_batch(X_testA)
    mfcc_trainB = extract_features_batch(X_trainB)
    mfcc_testB  = extract_features_batch(X_testB)

    # 3. Save Intermediate Results (For Part 2)
    print("\n3. Saving Intermediate Files ")
    
    # Save MFCCs
    np.save(f"{config.OUTPUT_DIR}/mfcc_trainA.npy", mfcc_trainA)
    np.save(f"{config.OUTPUT_DIR}/mfcc_testA.npy",  mfcc_testA)
    np.save(f"{config.OUTPUT_DIR}/mfcc_trainB.npy", mfcc_trainB)
    np.save(f"{config.OUTPUT_DIR}/mfcc_testB.npy",  mfcc_testB)

    # Save Labels
    np.save(f"{config.OUTPUT_DIR}/y_trainA_raw.npy", y_trainA)
    np.save(f"{config.OUTPUT_DIR}/y_testA_raw.npy",  y_testA)
    np.save(f"{config.OUTPUT_DIR}/y_trainB_raw.npy", y_trainB)
    np.save(f"{config.OUTPUT_DIR}/y_testB_raw.npy",  y_testB)

    print(f"\nAll MFCC arrays saved successfully to: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()