FROM python:3.11-slim


# Install system dependencies required by librosa / soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY api.py .
COPY inference.py .
COPY audio_processing.py .
COPY config.py .
COPY artifacts/ ./artifacts/


EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
