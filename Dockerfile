FROM python:3.10-slim

# System dependencies for WeasyPrint, spaCy, pyannote
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    ffmpeg \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY medai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Application code
COPY medai/ ./medai/
COPY medai/.env.example .env.example

# Start script
COPY docker-start.sh .
RUN chmod +x docker-start.sh

EXPOSE 8000 8501

CMD ["./docker-start.sh"]
