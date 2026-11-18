"""Report als Markdown: je Modell und Profil Score mit Konfidenzintervall, Tokens, Latenz, Kosten."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable

from harness_it_all.recorder import Record
from harness_it_all.scoring import Scored
from harness_it_all.stats import Estimate, block_bootstrap, paired_block_bootstrap

BlockFn = Callable[[Record], Hashable]


def group_key(r: Record) -> str:
    return f"{r.model} · {r.profile} ({r.profile_hash})"


def _fmt(e: Estimate) -> str:
    return f"{e.mean:+.4f} [{e.low:+.4f}, {e.high:+.4f}]"


def summarize(scored: list[Scored], block_of: BlockFn, exclude_fallbacks: bool = True) -> dict[str, dict]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for s in scored:
        groups[group_key(s.record)].append(s)
    out = {}
    for key, items in sorted(groups.items()):
        fallbacks = [s for s in items if (s.record.completion or {}).get("fallback_used")]
        use = [s for s in items if s.score is not None and not (exclude_fallbacks and s in fallbacks)]
        comps = [s.record.completion for s in items if s.record.completion]
        row = {
            "n_calls": len(items),
            "n_scored": len(use),
            "errors": sum(1 for s in items if s.record.error),
            "fallbacks": len(fallbacks),
            "latency_s": sum(c["latency_s"] for c in comps) / len(comps) if comps else None,
            "output_tokens": sum(c["output_tokens"] or 0 for c in comps) / len(comps) if comps else None,
            "cost_usd": sum(c.get("cost_usd") or 0 for c in comps),
            "energy_kwh": sum(c.get("energy_kwh") or 0 for c in comps),
            "asymmetries": items[0].record.asymmetries,
        }
        if use:
            row["score"] = block_bootstrap([s.score for s in use], [block_of(s.record) for s in use])
        out[key] = row
    return out


def compare(scored: list[Scored], a: str, b: str, block_of: BlockFn) -> Estimate:
    """Gepaarter Vergleich zweier Gruppen (Schlüssel wie in `summarize`) über gemeinsame Fälle."""
    by = {a: {}, b: {}}
    blocks = {}
    for s in scored:
        k = group_key(s.record)
        if k in by and s.score is not None and not (s.record.completion or {}).get("fallback_used"):
            by[k][s.record.case_id] = s.score
            blocks[s.record.case_id] = block_of(s.record)
    return paired_block_bootstrap(by[a], by[b], blocks)


def markdown(
    summary: dict[str, dict], title: str = "Ergebnis", comparisons: dict[str, Estimate] | None = None
) -> str:
    lines = [f"# {title}", "", "| Modell · Profil | Score (95 %-KI) | Fälle | Blöcke | Fehler | Fallbacks |"
             " Ø Latenz s | Ø Output-Tokens | Kosten $ | Strom kWh |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for key, r in summary.items():
        e = r.get("score")
        lines.append(
            f"| {key} | {_fmt(e) if e else '–'} | {r['n_scored']}/{r['n_calls']} | "
            f"{e.n_blocks if e else '–'} | "
            f"{r['errors']} | {r['fallbacks']} | {r['latency_s'] or 0:.2f} | {r['output_tokens'] or 0:.0f} | "
            f"{r['cost_usd']:.4f} | {r['energy_kwh']:.5f} |"
        )
    if comparisons:
        lines += ["", "## Gepaarte Vergleiche (Block-Bootstrap)", ""]
        for name, e in comparisons.items():
            verdict = "Unterschied belastbar" if e.excludes_zero else "kein belastbarer Unterschied"
            lines.append(f"- **{name}:** {_fmt(e)} · {e.n} Fälle in {e.n_blocks} Blöcken · {verdict}")
    notes = sorted({a for r in summary.values() for a in r["asymmetries"]})
    if notes:
        lines += ["", "## Bekannte Unterschiede der Einstellungen", ""] + [f"- {a}" for a in notes]
    lines += [
        "",
        "Fallback-Antworten (ein anderes Modell hat geantwortet) zählen nicht in Score und Vergleich.",
        "Weniger als etwa 8 Blöcke: Ergebnis nur vorläufig.",
    ]
    return "\n".join(lines) + "\n"
