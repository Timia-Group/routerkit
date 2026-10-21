#!/usr/bin/env python3
"""Edits connection fields (endpoint/api_key_env/deployment) of a single
already-active model in place, via PATCH /models/{model_id} — no cascade,
nothing else gets deactivated. Typically for fixing a `<...>` placeholder
left over from before realizing it was needed (e.g. right before trying
POST /route), without redoing a full catalog import (which would
deactivate solutions/targets/model_metrics/the active predictor just to
fix one field — see routers/catalogs.py's PATCH docstring).

`model_id` is the database's internal id, NOT the model's name — get it
from data/current/models.json (the live snapshot of what's actually
active; scripts/wizard.py already reads this for the same reason) rather
than guessing.

Usage:
    python scripts/edit_model.py <model_id> --endpoint=https://real-endpoint
    python scripts/edit_model.py <model_id> --api-key-env=MY_OTHER_KEY --deployment=gpt-4o-mini
"""

import argparse
import json
import urllib.error
import urllib.request


def _log(msg: str) -> None:
    print(f"[edit_model] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", type=int, help="Internal model_id (see data/current/models.json)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--endpoint", help="New endpoint/api_base URL")
    parser.add_argument("--api-key-env", help="New api_key_env (the .env variable name to read the key from)")
    parser.add_argument("--deployment", help="New deployment name")
    args = parser.parse_args()

    body = {
        k: v for k, v in {
            "endpoint": args.endpoint,
            "api_key_env": args.api_key_env,
            "deployment": args.deployment,
        }.items() if v is not None
    }
    if not body:
        parser.error("nothing to change — pass at least one of --endpoint/--api-key-env/--deployment")

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{args.base_url}/models/{args.model_id}", data=data, method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed local API
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[edit_model] HTTP {exc.code}: {detail}") from exc

    _log(f"updated model_id={result['model_id']} ({result['name']}, role={result['role']}):")
    for field in body:
        _log(f"  {field} = {result[field]!r}")


if __name__ == "__main__":
    main()
