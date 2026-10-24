"""RSNA knee MRI: shared names and small helpers.

LABELS is the order of the 12 findings everywhere (model outputs, label tables, submission columns).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

LABELS: tuple[str, ...] = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
                           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")

READ_BLOCK_BYTES = 1 << 20


def sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_bytes())


def write_json(path: str | Path, value, indent: int | None = 1) -> None:
    """Write JSON atomically (temporary file, then rename), so a reader never sees half a file."""
    path = Path(path)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=indent, allow_nan=False))
    os.replace(part, path)
