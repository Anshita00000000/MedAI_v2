FROM python:3.10-slim

# System dependencies for WeasyPrint, spaCy, pyannote, ffmpeg
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

# Install in two passes to avoid cross-package resolver backtracking:
# 1. Core deps (fast, no conflicts)
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.3" \
    streamlit==1.45.1 \
    jinja2==3.1.6 \
    python-dotenv==1.1.0 \
    pydantic==2.11.4 \
    requests==2.32.3 \
    beautifulsoup4==4.13.4 \
    "numpy>=1.26.0,<3.0.0" \
    pandas==2.2.3 \
    pypdf==5.5.0 \
    rouge-score==0.1.2 \
    bert-score==0.3.13 \
    jiwer==3.0.5 \
    google-genai==1.14.0 \
    weasyprint==62.3

# 2. ML / NLP deps (heavier, install separately to isolate resolver)
RUN pip install --no-cache-dir \
    "torch>=2.2.0,<3.0.0" \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    openai-whisper==20240930

RUN pip install --no-cache-dir \
    spacy==3.7.5 \
    scispacy==0.5.5 \
    medspacy==1.1.3

RUN pip install --no-cache-dir \
    sentence-transformers==3.4.1 \
    chromadb==0.6.3

RUN pip install --no-cache-dir \
    "pyannote.audio==3.3.2"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Application code
COPY medai/ ./medai/
COPY medai/.env.example .env.example

COPY docker-start.sh .
RUN chmod +x docker-start.sh

EXPOSE 8000 8501

CMD ["./docker-start.sh"]
