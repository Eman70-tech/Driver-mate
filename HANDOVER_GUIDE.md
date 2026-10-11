# Ford Engine Sound Classifier — Backend Handover Guide

> **Audience:** Backend engineers integrating or deploying this API.
> **Safety contract:** Faulty engines must never be silently cleared.
> A high threshold (0.847) is intentional — it is not a tuning parameter.

---

## 1. Files to Hand Over

```
Driver-mate/
│
├── api.py                        ← FastAPI application (entry point)
├── inference.py                  ← Core inference pipeline
├── audio_processing.py           ← MFCC extraction (do NOT modify)
├── config.py                     ← Signal-processing constants (do NOT modify)
├── requirements.txt
│
└── artifacts/                    ← AI model + preprocessing artifacts
    ├── best_sound_model.keras
    ├── optimal_threshold.npy
    └── scaler_FordA_mfcc.pkl
```

> **Critical:** `audio_processing.py` and `config.py` must be identical
> to the versions used during training. Never edit them independently.

---

## 2. How to Run the API Locally

### Prerequisites
- Python 3.10 or 3.11
- pip

### Steps

```bash
# 1. Clone / copy the project folder
cd Driver-mate

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## 3. Example Request / Response

### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@engine_recording.wav"
```

### Using Python (requests)
```python
import requests

url   = "http://localhost:8000/predict"
files = {"file": ("engine.wav", open("engine_recording.wav", "rb"), "audio/wav")}

response = requests.post(url, files=files)
print(response.json())
```

### Healthy Engine Response
```json
{
  "prediction":  "Healthy",
  "confidence":  0.91,
  "probability": 0.91,
  "threshold":   0.847,
  "message":     "No critical engine fault detected.",
  "chunk_votes": [1, 1, 1, 1, 1]
}
```

### Faulty Engine Response
```json
{
  "prediction":  "Faulty",
  "confidence":  0.88,
  "probability": 0.12,
  "threshold":   0.847,
  "message":     "Potential engine issue detected. Inspection recommended.",
  "chunk_votes": [0, 0, 1, 0, 0]
}
```

### Response Field Reference

| Field        | Type    | Description                                              |
|--------------|---------|----------------------------------------------------------|
| prediction   | string  | `"Healthy"` or `"Faulty"`                               |
| confidence   | float   | Certainty of the prediction (0–1)                       |
| probability  | float   | Raw P(Healthy) from the model — useful for logging      |
| threshold    | float   | Safety threshold in use (normally 0.847)                |
| message      | string  | Human-readable safety verdict                           |
| chunk_votes  | int[]   | Per-second vote: 1 = Healthy, 0 = Faulty                |

---

## 4. Required Dependencies

See `requirements.txt`. Key libraries:

| Library          | Purpose                                      |
|------------------|----------------------------------------------|
| fastapi          | REST API framework                           |
| uvicorn          | ASGI server                                  |
| tensorflow       | CNN model inference                          |
| librosa          | Audio loading + MFCC extraction              |
| scikit-learn     | StandardScaler (pre-fitted)                  |
| joblib           | Load the .pkl scaler file                    |
| python-multipart | File upload support in FastAPI               |

---

## 5. Expected Audio Format

| Property        | Value / Requirement                                     |
|-----------------|---------------------------------------------------------|
| Format          | **WAV** (`.wav`)                                        |
| Duration        | **5 seconds recommended** (shorter = zero-padded, longer = cropped) |
| Sample Rate     | Any — librosa resamples to 8 000 Hz automatically       |
| Channels        | Mono or stereo (librosa mixes down to mono)             |
| Bit Depth       | 16-bit PCM recommended                                  |
| Max File Size   | 50 MB (enforced by the API)                             |
| Recording tip   | Hold the phone near the engine; avoid covering the mic  |

---

## 6. Flutter Integration Notes

```dart
// Example multipart upload from Flutter
var request = http.MultipartRequest(
  'POST',
  Uri.parse('https://your-api-host/predict'),
);
request.files.add(
  await http.MultipartFile.fromPath('file', audioFilePath),
);
var response = await request.send();
var body = await response.stream.bytesToString();
var json = jsonDecode(body);

// Use json['prediction'] and json['message'] in the UI
```

---

## 7. Deployment Checklist

- [ ] Place all `artifacts/` files on the server before starting the API
- [ ] Confirm `MODEL_PATH`, `THRESHOLD_PATH`, `SCALER_PATH` in `inference.py` point to the correct locations
- [ ] Run `GET /health` after startup to confirm the model loaded (`"status": "ok"`)
- [ ] Set `--workers 1` in uvicorn (TensorFlow is not fork-safe; use multiple processes via gunicorn if scaling is needed)
- [ ] Use HTTPS in production
- [ ] Add authentication (API key or JWT) before exposing to the internet

---

## 8. Threshold Explanation (for the team)

The safety threshold is **0.847** (stored in `optimal_threshold.npy`).

- The model outputs **P(Healthy)** for each second of audio.
- Each 1-second chunk is scored independently: P(Healthy) **≥ 0.847** counts as a **Healthy** vote. The engine is cleared as **Healthy** only if `healthy_votes >= ceil(n_chunks × 0.80)` (4 of 5 chunks for a 5-second recording); otherwise it is reported as **Faulty**.

This high bar is deliberate: the model must be strongly confident an engine is healthy before clearing it. A healthy engine wrongly flagged as faulty (false alarm) is **acceptable**. A faulty engine wrongly cleared as healthy (missed fault) is **not acceptable**.

---

*Generated for the Ford Engine Sound Classification project · v1.0.0*
