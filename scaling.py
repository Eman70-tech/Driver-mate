import numpy as np
from sklearn.preprocessing import StandardScaler
import config

def fit_scaler(mfcc_original_train: np.ndarray) -> StandardScaler:

    N = mfcc_original_train.shape[0]
    #(Flattening)
    flat = mfcc_original_train.reshape(N, -1)  
    
    scaler = StandardScaler()
    scaler.fit(flat)
    
    return scaler

def apply_scaler(scaler: StandardScaler, mfcc_arr: np.ndarray) -> np.ndarray:

    N = mfcc_arr.shape[0]

    flat = mfcc_arr.reshape(N, -1)              # (N, 819)

    scaled = scaler.transform(flat)             # (N, 819)

    return scaled.reshape(N, config.N_MFCC, -1).astype(np.float32)