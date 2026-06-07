FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY gps_tracking/ ./gps_tracking/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

EXPOSE 8008

CMD ["python", "src/model_http_api.py"]
