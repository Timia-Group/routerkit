FROM python:3.11-slim

WORKDIR /app

# libgomp1: LightGBM's native library needs it (OpenMP runtime) — not
# present in the slim base image, fails at import time otherwise
# ("libgomp.so.1: cannot open shared object file") rather than at
# training time, so this is caught at build, not left for POST /train
# to discover.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download nltk's sentence tokenizer data at build time, so question
# scripts using nltk.tokenize.sent_tokenize work offline at runtime (no
# internet access needed inside the container, no first-request latency).
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab

COPY app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
