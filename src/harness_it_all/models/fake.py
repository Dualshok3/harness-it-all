"""Ein Fake-Modell für Tests und CI: deterministisch, ohne Netz."""

from __future__ import annotations

from collections.abc import Callable

from harness_it_all.models.base import Completion
from harness_it_all.profiles import Profile


class FakeModel:
    backend = "fake"

    def __init__(self, model: str = "fake", answer: Callable[[list[dict], dict | None], str] | None = None):
        self.model = model
        self.answer = answer or (lambda messages, schema: messages[-1]["content"])
        self.calls: list[list[dict]] = []

    def version(self) -> dict:
        return {"backend": self.backend, "model": self.model}

    def complete(self, messages: list[dict], profile: Profile, schema: dict | None = None) -> Completion:
        self.calls.append(messages)
        text = self.answer(messages, schema)
        n_in = sum(len(m["content"].split()) for m in messages)
        return Completion(
            text=text,
            model_requested=self.model,
            model_served=self.model,
            finish_reason="stop",
            input_tokens=n_in,
            output_tokens=len(text.split()),
            latency_s=0.001,
        )
