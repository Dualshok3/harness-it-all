"""Kanonisches JSON und Hashes, damit gleiche Eingaben überall denselben Hash bekommen."""

from __future__ import annotations

import hashlib
import json


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def short_hash(obj, n: int = 12) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()[:n]
