"""Feste Profile: wie ein Modell aufgerufen wird (Thinking, Effort, Sampling, max_tokens).

Ein Profil ist backend-übergreifend. Es legt für jedes Backend fest, was „gleich eingestellt“ bedeutet,
und hält die Unterschiede fest, die sich nicht vermeiden lassen (`asymmetries`).
Jeder Aufruf speichert den Hash des Profils. So sieht man später, ob zwei Läufe wirklich
unter denselben Einstellungen liefen.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from harness_it_all.canon import short_hash

DEFAULT_PROFILES = Path(__file__).resolve().parents[2] / "configs" / "profiles.toml"
BACKENDS = ("openai", "anthropic", "claude_code", "deepseek")
CLI_EFFORTS = ("low", "medium", "high", "max")  # claude -p --effort kennt kein xhigh
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    max_tokens: int
    backends: dict[str, dict] = field(default_factory=dict)
    description: str = ""
    asymmetries: tuple[str, ...] = ()

    def for_backend(self, backend: str) -> dict:
        if backend not in self.backends:
            raise ProfileError(f"Profil {self.name!r} hat keine Einstellungen für {backend!r}")
        return self.backends[backend]

    @property
    def hash(self) -> str:
        return short_hash({"name": self.name, "max_tokens": self.max_tokens, "backends": self.backends})


def _validate(p: Profile) -> None:
    a = p.backends.get("anthropic", {})
    if a:
        if a.get("effort") not in EFFORTS:
            raise ProfileError(f"{p.name}: anthropic.effort muss eines von {EFFORTS} sein")
        bad = {"temperature", "top_p", "top_k", "thinking", "budget_tokens"} & set(a)
        if bad:
            raise ProfileError(
                f"{p.name}: {sorted(bad)} gibt es für Claude Opus 5.5 nicht (Thinking ist immer an, "
                "Sampling wird abgelehnt). Gesteuert wird nur über effort."
            )
    c = p.backends.get("claude_code", {})
    if c and c.get("effort") not in CLI_EFFORTS:
        raise ProfileError(f"{p.name}: claude_code.effort muss eines von {CLI_EFFORTS} sein")
    bad = {"enable_thinking", "seed", "top_k"} & set(p.backends.get("deepseek", {}))
    if bad:
        raise ProfileError(f"{p.name}: {sorted(bad)} versteht die DeepSeek-API nicht")
    o = p.backends.get("openai", {})
    if o and "enable_thinking" in o and not isinstance(o["enable_thinking"], bool):
        raise ProfileError(f"{p.name}: openai.enable_thinking muss true oder false sein")


def load_profiles(path: Path | str = DEFAULT_PROFILES) -> dict[str, Profile]:
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, raw in d.get("profile", {}).items():
        backends = {b: dict(raw[b]) for b in BACKENDS if b in raw}
        p = Profile(
            name=name,
            max_tokens=int(raw["max_tokens"]),
            backends=backends,
            description=raw.get("description", ""),
            asymmetries=tuple(raw.get("asymmetries", ())),
        )
        _validate(p)
        out[name] = p
    if not out:
        raise ProfileError(f"{path}: keine Profile gefunden")
    return out
