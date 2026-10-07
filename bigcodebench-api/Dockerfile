# Reuses the official BigCodeBench image: it already ships the `bigcodebench`
# package (with the official dataset and the execution sandbox) and its heavy
# dependencies. We just add an HTTP API on top; the original entrypoint is
# replaced.
FROM bigcodebench/bigcodebench-evaluate:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# `python -m uvicorn` instead of the bare `uvicorn` command: this base image
# runs as a non-root user whose `pip install` lands console scripts under
# `~/.local/bin`, which isn't on PATH — invoking uvicorn as a module sidesteps
# that entirely (it only needs to be importable, not on PATH).
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
