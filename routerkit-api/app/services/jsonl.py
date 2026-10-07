"""Generic parsing of JSONL files (one JSON row per line) into a Pydantic model.

Used by every bulk load of data already computed externally (`type=load` in
tasks/import, features/compute, solutions/generate, solutions/validate):
same reason in every case — these lists scale to thousands of rows (one
task, solution, grade or feature per line), and JSONL allows reading/
generating/diffing them line by line without loading or reserializing the
entire file, unlike a single large JSON array.
"""

import json
from typing import TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_jsonl(raw: str, row_model: type[T]) -> list[T]:
    """Parses `raw` (contents of a .jsonl) into a list of `row_model`.

    400 if any line isn't valid JSON, 422 if it doesn't satisfy `row_model`.
    Empty lines are ignored.
    """
    rows: list[T] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"line {line_no}: invalid JSON ({exc})"
            ) from exc
        try:
            rows.append(row_model.model_validate(raw_row))
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "item": f"line {line_no}",
                    "errors": exc.errors(include_url=False, include_context=False),
                },
            ) from exc
    return rows
