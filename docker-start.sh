#!/bin/bash
set -e

# Start FastAPI in background
uvicorn medai.api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run medai/app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
