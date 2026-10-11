import numpy as np
import librosa
import config
import time


def augment_white_noise(audio: np.ndarray,
                        snr_db: float = config.AUG_NOISE_SNR_DB) -> np.ndarray:

    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0.0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def augment_time_stretch(audio: np.ndarray,
                         rate: float = None) -> np.ndarray:

    if rate is None:
        rate = np.random.uniform(config.AUG_STRETCH_MIN, config.AUG_STRETCH_MAX)

    stretched = librosa.effects.time_stretch(audio, rate=rate)

    if len(stretched) >= len(audio):
        return stretched[:len(audio)].astype(np.float32)
    else:
        pad = len(audio) - len(stretched)
        return np.pad(stretched, (0, pad)).astype(np.float32)


def augment_amplitude(audio: np.ndarray,
                      low:  float = config.AUG_AMP_LOW,
                      high: float = config.AUG_AMP_HIGH) -> np.ndarray:
    """
    تغيير مستوى الصوت (العلّو) بضرب الإشارة في رقم عشوائي.
    
    الهدف: محاكاة بعد الموبايل عن الموتور أو اختلاف حساسية المايك.
    """
    gain = np.random.uniform(low, high)
    return (audio * gain).astype(np.float32)



def augment_freq_mask(mfcc: np.ndarray,
                      F: int = config.AUG_FREQ_MASK_F) -> np.ndarray:

    mfcc_aug = mfcc.copy()
    # اختيار بداية الماسك بشكل عشوائي
    f_start = np.random.randint(0, mfcc.shape[0] - F)
    mfcc_aug[f_start : f_start + F, :] = 0.0
    return mfcc_aug.astype(np.float32)


def augment_batch_audio(X_raw: np.ndarray, aug_fn, split_label: str) -> np.ndarray:

    N = X_raw.shape[0]
    results = []
    t0 = time.time()
    print(f"    {split_label:<35}", end=" ", flush=True)

    for i in range(N):
        # 1. Resample to 8 kHz
        audio_8k = librosa.resample(X_raw[i].astype(np.float32),
                                    orig_sr=config.ORIG_SR, 
                                    target_sr=config.TARGET_SR)
        # 2. Augment in the audio domain
        audio_aug = aug_fn(audio_8k)
        
        # 3. Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio_aug, sr=config.TARGET_SR,
                                    n_mfcc=config.N_MFCC, n_fft=config.N_FFT,
                                    hop_length=config.HOP_LENGTH)
        results.append(mfcc)

    elapsed = time.time() - t0
    arr = np.stack(results, axis=0).astype(np.float32)
    print(f"→  {arr.shape}  ({elapsed:.1f}s)")
    return arr


def augment_batch_mfcc(mfcc_arr: np.ndarray, aug_fn, split_label: str) -> np.ndarray:

    print(f"    {split_label:<35}", end=" ", flush=True)
    t0 = time.time()
    results = np.stack([aug_fn(mfcc_arr[i]) for i in range(len(mfcc_arr))], axis=0)
    print(f"→  {results.shape}  ({time.time()-t0:.1f}s)")
    return results.astype(np.float32)