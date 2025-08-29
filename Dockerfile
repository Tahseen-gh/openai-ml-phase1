# FastAPI app container
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_app ./fastapi_app
COPY .env.example ./.env

EXPOSE 8000
CMD ["uvicorn", "fastapi_app.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
