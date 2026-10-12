#!/usr/bin/env python3
"""Calls POST /route with a single task prompt and prints the resulting
ranking. Stdlib only — no pip install needed.

Mirrors the endpoint exactly: POST /route takes exactly one argument
(`prompt`), so this script does too. It requires an active predictor
(POST /train, see scripts/run_pipeline.py) and an active model with
role='feature' that has a REAL endpoint configured — that's the only
model actually called here (to compute live features for the prompt);
candidate models are only scored, never called (see routers/routing.py).
Fixed to http://localhost:8000 — this project only ever runs locally via
docker compose, there's no other address to point it at.

Usage:
  python scripts/test_route.py "<task prompt>"
"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"


def _log(msg: str) -> None:
    print(f"[test_route] {msg}", flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"usage: {sys.argv[0]} \"<task prompt>\"")
        raise SystemExit(0 if len(sys.argv) == 2 else 2)
    prompt = sys.argv[1]

    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/route", data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    _log("calling POST /route ...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 - fixed local API
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} calling POST /route: {detail}") from exc

    _log(f"decision_id={result['decision_id']} lambda_used={result['lambda_used']:.4f} "
         f"features_cost_usd={result['features_cost_usd']:.4f}")
    _log("ranking:")
    for i, entry in enumerate(result["ranking"], start=1):
        print(
            f"  {i}. {entry['model_name']:<24} p={entry['p']:.4f}  "
            f"price={entry['price']:.6f}  score={entry['score']:.4f}"
        )


if __name__ == "__main__":
    main()
