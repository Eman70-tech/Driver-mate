import os
import logging
import numpy as np
import joblib
import tensorflow as tf


from audio_processing import process_live_audio_chunks

logger = logging.getLogger(__name__)


BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "artifacts", "best_sound_model.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "artifacts", "optimal_threshold.npy")
SCALER_PATH    = os.path.join(BASE_DIR, "artifacts", "scaler_FordA_mfcc.pkl")


FALLBACK_THRESHOLD = 0.847
HEALTHY_MAJORITY_RATIO = 0.80


class _InferenceEngine:


    _instance = None

    def __init__(self):
        logger.info("Loading model from: %s", MODEL_PATH)
        self.model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model input shape : %s", self.model.input_shape)

        logger.info("Loading scaler from: %s", SCALER_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        if os.path.exists(THRESHOLD_PATH):
            self.threshold = float(np.load(THRESHOLD_PATH))
            logger.info("Threshold loaded from file: %.4f", self.threshold)
        else:
            self.threshold = FALLBACK_THRESHOLD
            logger.warning(
                "optimal_threshold.npy not found — using fallback %.4f",
                self.threshold,
            )

    @classmethod
    def get(cls) -> "_InferenceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance



def predict_audio_file(file_path: str, duration: int = 5) -> dict:

    engine = _InferenceEngine.get()

    batch_mfccs = process_live_audio_chunks(file_path, duration=duration)
    

    n_chunks, H, W, C = batch_mfccs.shape

    flat = batch_mfccs.reshape(n_chunks, -1)          # (N, H*W*C)
    flat_scaled = engine.scaler.transform(flat)        # normalise

    X = flat_scaled.reshape(n_chunks, H, W, C)

    chunk_probs = engine.model.predict(X, verbose=0).flatten()   # (N,)
    logger.debug("Per-chunk P(Healthy): %s", chunk_probs)

    chunk_votes   = (chunk_probs >= engine.threshold).astype(int).tolist()
    healthy_votes = int(np.sum(chunk_votes))
    required_votes = max(1, int(np.ceil(n_chunks * HEALTHY_MAJORITY_RATIO)))
    is_healthy     = healthy_votes >= required_votes
    
    logger.info(
        "Votes: %d/%d healthy (required: %d) → %s",
        healthy_votes, n_chunks, required_votes,
        "Healthy" if is_healthy else "Faulty"
    )

    if is_healthy:
        prediction = "Healthy"
        confidence = healthy_votes / n_chunks
        severity   = "NONE"
        message    = "No critical engine fault detected."
    else:
        prediction = "Faulty"
        confidence = 1.0 - healthy_votes / n_chunks
        # Severity helps Flutter display the right UI urgency level
        faulty_votes = n_chunks - healthy_votes
        if faulty_votes == n_chunks:
            severity = "HIGH"    # all chunks voted Faulty — strong signal
        elif faulty_votes >= int(np.ceil(n_chunks * 0.6)):
            severity = "MEDIUM"  # clear majority, but some uncertainty
        else:
            severity = "LOW"     # marginal: just below healthy threshold
        message = "Potential engine issue detected. Inspection recommended."

    return {
        "prediction" : prediction,
        "confidence" : round(confidence, 4),
        "probability": round(healthy_votes / n_chunks, 4),  # raw P(Healthy) for logging
        "threshold"  : round(engine.threshold, 4),
        "severity"   : severity,   # "NONE" | "LOW" | "MEDIUM" | "HIGH"
        "message"    : message,
        "chunk_votes": chunk_votes,    # e.g. [1, 1, 0, 1, 1]
    }