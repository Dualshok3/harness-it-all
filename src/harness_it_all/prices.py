"""Kosten: API-Preise für Claude, Stromschätzung für lokale Modelle (`configs/prices.toml`)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PRICES = Path(__file__).resolve().parents[2] / "configs" / "prices.toml"


@dataclass(frozen=True)
class Prices:
    api: dict[str, tuple[float, float]] = field(default_factory=dict)  # Modell → (Input, Output) $/MTok
    gpu_watts: float = 0.0
    eur_per_kwh: float = 0.0

    def api_cost(self, model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if model not in self.api or input_tokens is None or output_tokens is None:
            return None
        inp, out = self.api[model]
        return (input_tokens * inp + output_tokens * out) / 1_000_000

    def energy_kwh(self, seconds: float) -> float:
        return self.gpu_watts * seconds / 3600 / 1000


def load_prices(path: Path | str = DEFAULT_PRICES) -> Prices:
    d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    api = {m: (v["input_usd_per_mtok"], v["output_usd_per_mtok"]) for m, v in d.get("api", {}).items()}
    local = d.get("local", {})
    return Prices(api, local.get("gpu_watts", 0.0), local.get("electricity_eur_per_kwh", 0.0))
