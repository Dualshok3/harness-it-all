"""Aufzeichnen, Ausführen und Wiederholen von Modellaufrufen.

Jeder Aufruf wird als eine JSON-Zeile festgehalten: Fall, Eingabe-Hash, Profil samt Hash, Prompt-Version,
Modell- und Engine-Version, Antwort, Tokens, Latenz, Kosten und frei wählbare Tags (z. B. der Snapshot-Hash
im Trading-Projekt). Mit `store_inputs=True` (Standard) wird auch die Eingabe gespeichert, damit sich jeder
Aufruf später mit einem anderen Modell oder Profil wiederholen lässt (Replay).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from harness_it_all.canon import short_hash
from harness_it_all.models.base import ChatModel, ModelError
from harness_it_all.profiles import Profile


@dataclass
class Record:
    case_id: str
    input_hash: str
    model: str
    backend: str
    profile: str
    profile_hash: str
    prompt_version: str
    at: str
    tags: dict = field(default_factory=dict)
    model_version: dict = field(default_factory=dict)
    asymmetries: list = field(default_factory=list)
    messages: list | None = None
    schema: dict | None = None
    completion: dict | None = None
    error: str | None = None
    replay_of: str | None = None

    @property
    def id(self) -> str:
        return short_hash([self.case_id, self.input_hash, self.model, self.profile_hash, self.at], 16)

    def to_dict(self) -> dict:
        return asdict(self) | {"id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> Record:
        d = dict(d)
        d.pop("id", None)
        return cls(**d)


def input_hash(messages: list[dict], schema: dict | None) -> str:
    return short_hash({"messages": messages, "schema": schema}, 16)


class Recorder:
    def __init__(self, path: Path | str, store_inputs: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_inputs = store_inputs

    def write(self, record: Record) -> Record:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def read(self) -> list[Record]:
        return list(read_records(self.path))

    def run(
        self,
        model: ChatModel,
        profile: Profile,
        case_id: str,
        messages: list[dict],
        schema: dict | None = None,
        prompt_version: str = "",
        tags: dict | None = None,
        replay_of: str | None = None,
    ) -> Record:
        """Ein Modell aufrufen und den Aufruf aufzeichnen. Fehler werden aufgezeichnet, nicht verschluckt."""
        rec = Record(
            case_id=case_id,
            input_hash=input_hash(messages, schema),
            model=model.model,
            backend=model.backend,
            profile=profile.name,
            profile_hash=profile.hash,
            prompt_version=prompt_version,
            at=datetime.now(UTC).isoformat(),
            tags=dict(tags or {}),
            model_version=model.version(),
            asymmetries=list(profile.asymmetries),
            messages=messages if self.store_inputs else None,
            schema=schema if self.store_inputs else None,
            replay_of=replay_of,
        )
        try:
            rec.completion = model.complete(messages, profile, schema).to_dict()
        except ModelError as e:
            rec.error = str(e)
        return self.write(rec)

    def replay(
        self, records: Iterable[Record], model: ChatModel, profile: Profile, prompt_version: str | None = None
    ) -> list[Record]:
        """Aufgezeichnete Eingaben mit einem anderen Modell oder Profil erneut ausführen."""
        out, seen = [], set()
        for r in records:
            if r.messages is None:
                raise ValueError(f"{r.case_id}: Eingabe nicht gespeichert, Replay unmöglich")
            key = (r.case_id, r.input_hash)
            if key in seen:
                continue  # dieselbe Eingabe nur einmal wiederholen
            seen.add(key)
            out.append(
                self.run(
                    model,
                    profile,
                    r.case_id,
                    r.messages,
                    r.schema,
                    prompt_version if prompt_version is not None else r.prompt_version,
                    r.tags,
                    replay_of=r.id,
                )
            )
        return out


def read_records(path: Path | str) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.from_dict(json.loads(line))
