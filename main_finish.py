import numpy as np
import joblib
import config
from data_loader import load_dataset
from scaling import fit_scaler, apply_scaler

def encode_labels(y: np.ndarray) -> np.ndarray:

    return ((y + 1) // 2).astype(np.int32)

def final_reshape(X: np.ndarray) -> np.ndarray:

    return X[..., np.newaxis]

def main():

    print(" FINAL PHASE: SCALING, ENCODING & RESHAPING")

    mfcc_trainA_orig = np.load(f"{config.OUTPUT_DIR}/mfcc_trainA.npy")
    mfcc_trainB_orig = np.load(f"{config.OUTPUT_DIR}/mfcc_trainB.npy")
    mfcc_testA_orig  = np.load(f"{config.OUTPUT_DIR}/mfcc_testA.npy")
    mfcc_testB_orig  = np.load(f"{config.OUTPUT_DIR}/mfcc_testB.npy")

    _, _, _, y_testA_raw = load_dataset(config.DATA_DIR, "FordA")
    _, _, _, y_testB_raw = load_dataset(config.DATA_DIR, "FordB")

    mfcc_trainA_aug = np.load(f"{config.OUTPUT_DIR}/mfcc_trainA_aug.npy")
    y_trainA_aug    = np.load(f"{config.OUTPUT_DIR}/y_trainA_aug.npy")
    
    mfcc_trainB_aug = np.load(f"{config.OUTPUT_DIR}/mfcc_trainB_aug.npy")
    y_trainB_aug    = np.load(f"{config.OUTPUT_DIR}/y_trainB_aug.npy")

    print("\nFitting Scalers on Original Data")
    scaler_A = fit_scaler(mfcc_trainA_orig)
    scaler_B = fit_scaler(mfcc_trainB_orig)

    print("Applying Scalers to Augmented and Test Data")
    # توحيد قياسي لـ FordA
    trainA_scaled = apply_scaler(scaler_A, mfcc_trainA_aug)
    testA_scaled  = apply_scaler(scaler_A, mfcc_testA_orig)
    # توحيد قياسي لـ FordB
    trainB_scaled = apply_scaler(scaler_B, mfcc_trainB_aug)
    testB_scaled  = apply_scaler(scaler_B, mfcc_testB_orig)

    print("Encoding Labels & Final Reshaping")
    # Label Encoding (0 and 1)
    y_trainA_final = encode_labels(y_trainA_aug)
    y_testA_final  = encode_labels(y_testA_raw)
    y_trainB_final = encode_labels(y_trainB_aug)
    y_testB_final  = encode_labels(y_testB_raw)

    # Reshaping (Adding Channel Dimension)
    X_trainA_final = final_reshape(trainA_scaled)
    X_testA_final  = final_reshape(testA_scaled)
    X_trainB_final = final_reshape(trainB_scaled)
    X_testB_final  = final_reshape(testB_scaled)

    print("\nSAVING FINAL PRODUCTION ARTEFACTS (SECTION 6)")
    

    np.save(f"{config.OUTPUT_DIR}/final_X_trainA.npy", X_trainA_final)
    np.save(f"{config.OUTPUT_DIR}/final_y_trainA.npy", y_trainA_final)
    np.save(f"{config.OUTPUT_DIR}/final_X_testA.npy",  X_testA_final)
    np.save(f"{config.OUTPUT_DIR}/final_y_testA.npy",  y_testA_final)

    np.save(f"{config.OUTPUT_DIR}/final_X_trainB.npy", X_trainB_final)
    np.save(f"{config.OUTPUT_DIR}/final_y_trainB.npy", y_trainB_final)
    np.save(f"{config.OUTPUT_DIR}/final_X_testB.npy",  X_testB_final)
    np.save(f"{config.OUTPUT_DIR}/final_y_testB.npy",  y_testB_final)

    joblib.dump(scaler_A, f"{config.OUTPUT_DIR}/scaler_FordA_mfcc.pkl")
    joblib.dump(scaler_B, f"{config.OUTPUT_DIR}/scaler_FordB_mfcc.pkl")

    print(f" DL-ready tensors saved in '{config.OUTPUT_DIR}'")
    print(f"   final_X_trainA shape: {X_trainA_final.shape}")
    print(f"   final_X_trainB shape: {X_trainB_final.shape}")



if __name__ == "__main__":
    main()