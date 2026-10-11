import numpy as np
import librosa
import config


def process_single_audio(raw_signal):

    audio = np.array(raw_signal, dtype=np.float32)

    # 2. Resample (500 Hz -> 8000 Hz)
    audio_8k = librosa.resample(
        y=audio,
        orig_sr=config.ORIG_SR,
        target_sr=config.TARGET_SR
    )

    # 3. Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio_8k,
        sr=config.TARGET_SR,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    
    return mfccs


def extract_features_batch(X_data):

    print(f"Processing {len(X_data)} samples for MFCC extraction.")

    mfcc_list = [process_single_audio(sample) for sample in X_data]
        
    return np.stack(mfcc_list, axis=0) # Final Shape: (N, 13, 63)



""" PRODUCTION LIVE AUDIO FUNCTION (BACKEND) """


def process_live_audio_chunks(file_path, duration=5):

    print(f"Processing live audio from mobile: {file_path}")
    
    # 1. Load audio and resample to target sample rate (8000 Hz)
    signal, sr = librosa.load(file_path, sr=config.TARGET_SR)
    

    target_total_length = config.TARGET_SR * duration # e.g., 8000 * 5 = 40,000 samples
    current_length = len(signal)
    
    if current_length > target_total_length:
        # If longer than duration, crop the excess from the end
        signal = signal[:target_total_length]
    elif current_length < target_total_length:
        # If shorter than duration, pad with zeros (silence)
        pad_length = target_total_length - current_length
        signal = np.pad(signal, (0, pad_length), mode='constant')


    chunks = np.split(signal, duration)

    mfccs_list = []
    for i, chunk in enumerate(chunks):
        mfcc = librosa.feature.mfcc(
            y=chunk, 
            sr=sr, 
            n_mfcc=config.N_MFCC, 
            n_fft=config.N_FFT, 
            hop_length=config.HOP_LENGTH
        )
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfccs_list.append(mfcc)
    

    batch_mfccs = np.array(mfccs_list)
    
    print(f"Data prepared, Final shape: {batch_mfccs.shape}")
    # MFCC Shape: (5, 13, 63, 1) -> (Chunks, Height, Width, Channels)
    
    return batch_mfccs




