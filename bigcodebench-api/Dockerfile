# Reuses the official BigCodeBench image: it already ships the `bigcodebench`
# package (with the official dataset and the execution sandbox) and its heavy
# dependencies. We just add an HTTP API on top; the original entrypoint is
# replaced.
#
# Pinned to v0.2.1.post6 (~9.5GB) instead of `latest` (~25GB): `latest`'s
# extra weight is almost entirely generation-side provider SDKs (openai,
# anthropic, transformers, vllm, ...) we never import — we only use
# bigcodebench.data/eval/sanitize (dataset lookup + the sandboxed test
# runner), never bigcodebench.evaluate/generate. Verified against our own
# real experiment data (solutions_exp.jsonl vs validations_exp.jsonl,
# ~39k real comparisons): 99%+ exact match; the ~7 tasks that didn't match
# are excluded from tasks_exp.jsonl/features_exp.jsonl/solutions_exp.jsonl/
# validations_exp.jsonl for this reason (see data/examples/README.md).
FROM bigcodebench/bigcodebench-evaluate:v0.2.1.post6

WORKDIR /app

COPY requirements.txt .
# tree_sitter_python (needed by bigcodebench.sanitize, which this image
# doesn't ship) pulls in a newer tree_sitter than the one already pinned
# here — --upgrade forces it, otherwise pip silently keeps the
# incompatible older one and sanitize() fails at import time.
RUN pip install --no-cache-dir -r requirements.txt --upgrade "tree_sitter>=0.22" tree_sitter_python

COPY app ./app

EXPOSE 8000

# `python -m uvicorn` instead of the bare `uvicorn` command: this base image
# runs as a non-root user whose `pip install` lands console scripts under
# `~/.local/bin`, which isn't on PATH — invoking uvicorn as a module sidesteps
# that entirely (it only needs to be importable, not on PATH).
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
