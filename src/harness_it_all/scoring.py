"""Bewertung: Die Harness kennt keine Fachlogik. Ein Projekt liefert einen Scorer.

Ein Scorer bekommt eine `Record` und gibt eine Zahl zurück (höher ist besser) oder `None`, wenn der Fall nicht
bewertbar ist (z. B. weil die Kursentwicklung danach noch fehlt). Fehlgeschlagene Aufrufe bekommen den Wert
`on_error`, damit Fehler nicht stillschweigend aus der Wertung fallen.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from harness_it_all.recorder import Record

Scorer = Callable[[Record], float | None]


@dataclass(frozen=True)
class Scored:
    record: Record
    score: float | None


def score_records(records: Iterable[Record], scorer: Scorer, on_error: float | None = 0.0) -> list[Scored]:
    out = []
    for r in records:
        if r.error is not None or r.completion is None:
            out.append(Scored(r, on_error))
        else:
            out.append(Scored(r, scorer(r)))
    return out


def exact_match(expected: dict[str, str]) -> Scorer:
    """1, wenn der Text (ohne Leerraum am Rand) der erwarteten Antwort des Falls entspricht, sonst 0."""

    def score(r: Record) -> float | None:
        if r.case_id not in expected:
            return None
        return float(r.completion["text"].strip() == expected[r.case_id].strip())

    return score


def json_valid(required: Iterable[str] = ()) -> Scorer:
    """1, wenn die Antwort gültiges JSON ist und alle Pflichtfelder enthält, sonst 0."""
    keys = tuple(required)

    def score(r: Record) -> float:
        try:
            obj = json.loads(r.completion["text"])
        except (json.JSONDecodeError, TypeError):
            return 0.0
        return float(isinstance(obj, dict) and all(k in obj for k in keys))

    return score
