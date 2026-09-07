FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by vision libraries
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render exposes $PORT dynamically; default fallback to 7860 for local testing
ENV PORT=7860
EXPOSE $PORT

CMD ["python", "app.py"]