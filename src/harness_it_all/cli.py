"""Kommandozeile: Profile anzeigen und einen Lauf gegen die lokale Engine oder Claude prüfen."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from harness_it_all.profiles import DEFAULT_PROFILES, load_profiles

app = typer.Typer(no_args_is_help=True, help="harness-it-all: faire Vergleiche von LLM-Agenten.")
console = Console()


@app.command()
def profiles(path: Path = typer.Option(DEFAULT_PROFILES, help="Profil-Datei (TOML)")) -> None:
    """Alle Profile mit Hash, Einstellungen je Backend und bekannten Unterschieden."""
    for p in load_profiles(path).values():
        console.print(f"[bold]{p.name}[/bold] ({p.hash}) · max_tokens {p.max_tokens} · {p.description}")
        for backend, cfg in p.backends.items():
            console.print(f"  {backend}: {cfg}")
        for a in p.asymmetries:
            console.print(f"  [yellow]Unterschied:[/yellow] {a}")


@app.command()
def ping(
    backend: str = typer.Argument(..., help="openai oder anthropic"),
    model: str = typer.Option(None, help="Modell-ID (crit: exakt, z. B. run-…/candidate-0017)"),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="nur openai"),
    profile: str = typer.Option("minimal", help="Profilname"),
    out: Path = typer.Option(Path("runs") / "ping.jsonl", help="Aufzeichnung"),
) -> None:
    """Ein einzelner, aufgezeichneter Testaufruf. Bei anthropic kostet das einen API-Aufruf."""
    from harness_it_all.prices import load_prices
    from harness_it_all.recorder import Recorder

    prices = load_prices()
    if backend == "openai":
        from harness_it_all.models.openai_compat import OpenAICompatModel

        if not model:
            raise typer.BadParameter("--model ist für openai Pflicht (exakte Modell-ID der Engine)")
        m = OpenAICompatModel(model, base_url, prices)
    elif backend == "anthropic":
        from harness_it_all.models.anthropic_model import AnthropicModel

        m = AnthropicModel(model or "claude-opus-5-5", prices)
    else:
        raise typer.BadParameter("backend muss openai oder anthropic sein")
    p = load_profiles()[profile]
    rec = Recorder(out).run(m, p, "ping", [{"role": "user", "content": "Antworte nur mit: ok"}])
    if rec.error:
        console.print(f"[red]Fehler:[/red] {rec.error}")
        raise typer.Exit(1)
    c = rec.completion
    console.print(
        f"{c['model_served']}: {c['text']!r} · {c['latency_s']:.2f} s · {c['output_tokens']} Tokens · "
        f"Fallback: {c['fallback_used']}"
    )


if __name__ == "__main__":
    app()
