"""Gemeinsame Schnittstelle aller Modelle.

Nachrichten sind schlicht: `[{"role": "system" | "user" | "assistant", "content": str}]`.
Jeder Adapter übersetzt sie in das Format seiner API.
Eine optionale JSON-Schema-Vorgabe (`schema`) erzwingt strukturierte Ausgabe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from harness_it_all.profiles import Profile


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    model_requested: str
    model_served: str  # kann abweichen, z. B. bei einem serverseitigen Fallback
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    reasoning: str | None = None
    cost_usd: float | None = None  # API-Kosten
    energy_kwh: float | None = None  # lokale Modelle: grob geschätzter Strom
    fallback_used: bool = False
    extra: dict = field(default_factory=dict)  # backend-spezifisch, z. B. crit-Telemetrie

    def to_dict(self) -> dict:
        return asdict(self)


class ChatModel(Protocol):
    backend: str  # "openai" | "anthropic" | "fake"
    model: str

    def version(self) -> dict:
        """Was genau antwortet: Modell-ID, Engine/Kandidat, SDK-Version …"""
        ...

    def complete(self, messages: list[dict], profile: Profile, schema: dict | None = None) -> Completion: ...


def split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """System-Nachrichten am Anfang zusammenfassen und vom Rest trennen."""
    system, rest = [], []
    for m in messages:
        if m["role"] == "system" and not rest:
            system.append(m["content"])
        else:
            rest.append(m)
    if any(m["role"] == "system" for m in rest):
        raise ModelError("System-Nachrichten sind nur am Anfang erlaubt")
    return ("\n\n".join(system) if system else None), rest
