"""Statistik für abhängige Beobachtungen: Block-Bootstrap, auch gepaart.

Entscheidungen eines Tages oder einer Woche hängen zusammen (gleiche Nachrichtenlage, überlappende
Bewertungsfenster). Ein Bootstrap über einzelne Werte macht die Konfidenzintervalle dann zu eng. Hier werden
deshalb ganze Blöcke gezogen. Welche Beobachtung zu welchem Block gehört, legt das aufrufende Projekt fest
(z. B. Handelswoche).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int
    n_blocks: int
    level: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "n_blocks": self.n_blocks,
            "level": self.level,
        }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def block_bootstrap(
    values: list[float],
    blocks: list[Hashable],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Mittelwert mit Konfidenzintervall, gezogen werden ganze Blöcke mit Zurücklegen."""
    if len(values) != len(blocks):
        raise ValueError("values und blocks müssen gleich lang sein")
    if not values:
        raise ValueError("keine Werte")
    groups: dict[Hashable, list[float]] = defaultdict(list)
    for v, b in zip(values, blocks, strict=True):
        groups[b].append(v)
    keys = sorted(groups, key=str)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            total += sum(g)
            count += len(g)
        means.append(total / count)
    means.sort()
    alpha = (1 - level) / 2
    return Estimate(
        mean=sum(values) / len(values),
        low=_quantile(means, alpha),
        high=_quantile(means, 1 - alpha),
        n=len(values),
        n_blocks=len(keys),
        level=level,
    )


def paired_block_bootstrap(
    a: dict[Hashable, float],
    b: dict[Hashable, float],
    block_of: dict[Hashable, Hashable],
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Differenz A − B über gemeinsame Fälle. Ein Block enthält immer beide Seiten derselben Fälle."""
    common = sorted(set(a) & set(b) & set(block_of), key=str)
    if not common:
        raise ValueError("keine gemeinsamen Fälle")
    diffs = [a[k] - b[k] for k in common]
    return block_bootstrap(diffs, [block_of[k] for k in common], n_boot, level, seed)
