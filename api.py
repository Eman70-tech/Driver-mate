import os
import uuid
import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import predict_audio_file, _InferenceEngine

from fastapi.middleware.cors import CORSMiddleware



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {".wav"}
MAX_FILE_SIZE_MB   = 50  # reject suspiciously large uploads early



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Ford Engine API starting up")
    try:
        _InferenceEngine.get()           # loads model, scaler, threshold once
        logger.info("Model warm-up complete. Ready to serve requests.")
    except Exception as exc:
        logger.error("STARTUP FAILED: %s", exc)
        raise
    yield
    logger.info("Ford Engine API shutting down")



app = FastAPI(
    title       = "Ford Engine Sound Classifier",
    description = (
        "Safety-oriented AI backend that detects engine faults from audio recordings.\n\n"
        "**Safety contract**: the threshold is tuned so that *missed faults* "
        "(a faulty engine predicted as healthy) are minimised. "
        "False alarms (a healthy engine flagged as faulty) are acceptable."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    prediction : str    # "Healthy" | "Faulty"
    confidence : float  # certainty of the prediction  (0–1)
    threshold  : float  # safety threshold in use
    message    : str    # human-readable verdict

class DetailedPredictionResponse(PredictionResponse):
    """Extended response including raw model output — useful for debugging."""
    probability : float         # raw P(Healthy) from the model
    chunk_votes : list[int]     # per-second vote: 1=Healthy, 0=Faulty
    severity    : str



@app.get("/", include_in_schema=False)
async def root():
    return {
        "service"  : "Ford Engine Sound Classifier",
        "version"  : "1.0.0",
        "endpoints": {
            "POST /predict": "Upload a .wav file and receive a safety verdict",
            "GET  /health" : "Liveness probe",
            "GET  /docs"   : "Interactive Swagger UI",
        },
    }


@app.get("/health", tags=["System"])
async def health():

    engine = _InferenceEngine.get()
    return {
        "status"   : "ok",
        "threshold": round(engine.threshold, 4),
    }


@app.post(
    "/predict",
    response_model = DetailedPredictionResponse,
    tags           = ["Inference"],
    summary        = "Classify engine health from a .wav recording",
    responses      = {
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid file (wrong format, too large, etc.)"},
        422: {"description": "Validation error"},
        500: {"description": "Internal inference error"},
    },
)
async def predict(
    file: UploadFile = File(
        ...,
        description=(
            "Engine audio recording in WAV format. "
            "5 seconds is recommended; shorter clips are zero-padded, longer ones are cropped."
        ),
    ),
):
  

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No file name provided.")

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only {ALLOWED_EXTENSIONS} are accepted.",
        )


   

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    logger.info("Received file: %s  (%.2f MB)", file.filename, size_mb)


 
    tmp_suffix = f"_{uuid.uuid4().hex}.wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Saved to temp path: %s", tmp_path)

    
       
    
        result = predict_audio_file(tmp_path, duration=5)

    except HTTPException:
        raise   

    except Exception as exc:
        logger.exception("Inference failed for file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Ensure the uploaded file is a valid WAV recording.",
        )

    finally:
       
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Temp file removed: %s", tmp_path)


    

    logger.info(
        "Result → prediction=%s  confidence=%.4f  probability=%.4f  votes=%s",
        result["prediction"],
        result["confidence"],
        result["probability"],
        result["chunk_votes"],
    )
    return result
