"""Validated JSONL, CSV, and Parquet readers with no silent row dropping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from embedding_model.exceptions import DataValidationError

RecordT = TypeVar("RecordT", bound=BaseModel)
SUPPORTED_SUFFIXES = {".jsonl", ".csv", ".parquet"}


def read_records(path: str | Path, record_type: type[RecordT]) -> list[RecordT]:
    """Read every row, validate it, and reject duplicate non-null record IDs."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise DataValidationError(f"data file does not exist: {input_path}")
    if input_path.suffix not in SUPPORTED_SUFFIXES:
        raise DataValidationError(
            f"unsupported data extension {input_path.suffix!r}; expected JSONL, CSV, or Parquet"
        )
    rows: Iterable[dict[str, object]]
    if input_path.suffix == ".jsonl":
        rows = _read_jsonl(input_path)
    elif input_path.suffix == ".csv":
        rows = pd.read_csv(input_path).to_dict(orient="records")
    else:
        try:
            rows = pd.read_parquet(input_path).to_dict(orient="records")
        except ImportError as exc:
            raise DataValidationError(
                "Parquet input requires the optional 'parquet' dependencies"
            ) from exc
    records: list[RecordT] = []
    for row_number, row in enumerate(rows, start=1):
        try:
            records.append(record_type.model_validate(row))
        except ValidationError as exc:
            raise DataValidationError(f"invalid row {row_number} in {input_path}: {exc}") from exc
    _reject_duplicate_ids(records, input_path)
    if not records:
        raise DataValidationError(f"data file is empty: {input_path}")
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise DataValidationError(f"blank JSONL line {line_number} in {path}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"invalid JSON on line {line_number} in {path}") from exc
        if not isinstance(row, dict):
            raise DataValidationError(f"JSONL line {line_number} must contain an object")
        rows.append(row)
    return rows


def _reject_duplicate_ids(records: Sequence[BaseModel], path: Path) -> None:
    seen: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        record_id = getattr(record, "record_id", None)
        if record_id is not None:
            if record_id in seen:
                raise DataValidationError(
                    f"duplicate record_id {record_id!r} at row {row_number} in {path}"
                )
            seen.add(record_id)
