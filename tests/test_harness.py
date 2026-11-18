import json
from types import SimpleNamespace

import httpx
import pytest

from harness_it_all.models.anthropic_model import FALLBACK_BETA, AnthropicModel
from harness_it_all.models.base import ModelError, split_system
from harness_it_all.models.fake import FakeModel
from harness_it_all.models.openai_compat import OpenAICompatModel
from harness_it_all.prices import Prices, load_prices
from harness_it_all.profiles import Profile, ProfileError, load_profiles
from harness_it_all.recorder import Recorder, read_records
from harness_it_all.report import compare, markdown, summarize
from harness_it_all.scoring import exact_match, json_valid, score_records
from harness_it_all.stats import block_bootstrap, paired_block_bootstrap

PROFILES = load_profiles()
MSGS = [{"role": "system", "content": "Sei knapp."}, {"role": "user", "content": "Sag ok"}]


# ------------------------------------------------------------------------------------------------ Profile


def test_shipped_profiles():
    assert {"minimal", "think", "vote"} <= set(PROFILES)
    m = PROFILES["minimal"]
    assert m.for_backend("openai")["enable_thinking"] is False
    assert m.for_backend("anthropic")["effort"] == "low"
    assert m.asymmetries and len(m.hash) == 12


def test_profile_hash_changes_with_settings():
    a = Profile("x", 100, {"openai": {"temperature": 0.0}})
    b = Profile("x", 100, {"openai": {"temperature": 0.1}})
    assert a.hash != b.hash and a.hash == Profile("x", 100, {"openai": {"temperature": 0.0}}).hash


def test_opus_profile_rejects_sampling_and_thinking(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text(
        '[profile.bad]\nmax_tokens = 10\n[profile.bad.anthropic]\neffort = "low"\ntemperature = 0.2\n'
    )
    with pytest.raises(ProfileError, match="effort"):
        load_profiles(p)
    p.write_text('[profile.bad]\nmax_tokens = 10\n[profile.bad.anthropic]\neffort = "huge"\n')
    with pytest.raises(ProfileError):
        load_profiles(p)


def test_split_system():
    system, rest = split_system(MSGS)
    assert system == "Sei knapp." and rest == [MSGS[1]]
    with pytest.raises(ModelError):
        split_system([MSGS[1], MSGS[0]])


# ------------------------------------------------------------------------------------------------ Adapter


def test_openai_compat_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "run-x/candidate-0017",
                "choices": [
                    {"message": {"content": "ok", "reasoning_content": None}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                "crit_telemetry": {"decode_tps": 101.0},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    m = OpenAICompatModel("run-x/candidate-0017", prices=Prices(gpu_watts=180), client=client)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
              "additionalProperties": False}  # fmt: skip
    c = m.complete(MSGS, PROFILES["minimal"], schema)
    body = seen["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0 and body["seed"] == 7 and body["max_tokens"] == 4000
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert c.text == "ok" and c.output_tokens == 1 and c.extra["crit_telemetry"]["decode_tps"] == 101.0
    assert c.energy_kwh is not None and c.cost_usd is None


def test_openai_compat_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad schema")))
    with pytest.raises(ModelError, match="400"):
        OpenAICompatModel("m", client=client).complete(MSGS, PROFILES["minimal"])


class FakeAnthropicClient:
    def __init__(self, model_served="claude-opus-5-5", iterations=None):
        self.kwargs = None
        self.model_served = model_served
        self.iterations = iterations or []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model=self.model_served,
            stop_reason="end_turn",
            stop_details=None,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500, iterations=self.iterations),
        )


def test_anthropic_request_never_sends_sampling_or_disabled_thinking():
    client = FakeAnthropicClient()
    prices = Prices(api={"claude-opus-5-5": (4.0, 20.0)})
    m = AnthropicModel(prices=prices, client=client)
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    c = m.complete(MSGS, PROFILES["minimal"], schema)
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == "Sei knapp." and kw["messages"] == [MSGS[1]]
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(kw)
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert c.text == "ok" and c.reasoning is None
    assert c.cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert c.fallback_used is False


def test_anthropic_fallback_is_detected():
    client = FakeAnthropicClient(
        model_served="claude-opus-4-8", iterations=[SimpleNamespace(type="fallback_message")]
    )
    c = AnthropicModel(client=client).complete(MSGS, PROFILES["minimal"])
    assert c.fallback_used and c.model_served == "claude-opus-4-8"


def test_anthropic_without_fallbacks_uses_plain_endpoint():
    client = FakeAnthropicClient()
    AnthropicModel(client=client, fallbacks=False).complete(MSGS, PROFILES["minimal"])
    assert "betas" not in client.kwargs and "fallbacks" not in client.kwargs


def test_shipped_prices():
    p = load_prices()
    assert p.api["claude-opus-5-5"] == (4.0, 20.0)
    assert p.energy_kwh(3600) == pytest.approx(0.18)


# ----------------------------------------------------------------------------------------- Recorder/Replay


def test_record_and_replay(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    a = FakeModel("a")
    r1 = rec.run(a, PROFILES["minimal"], "c1", MSGS, prompt_version="v1", tags={"snapshot": "abc"})
    rec.run(a, PROFILES["minimal"], "c1", MSGS, prompt_version="v1")  # gleiche Eingabe nochmal
    assert r1.completion["text"] == "Sag ok" and r1.profile_hash == PROFILES["minimal"].hash
    b = FakeModel("b", answer=lambda m, s: "anders")
    replays = rec.replay(rec.read(), b, PROFILES["think"])
    assert len(replays) == 1  # dieselbe Eingabe nur einmal
    assert replays[0].replay_of == r1.id and replays[0].tags == {"snapshot": "abc"}
    assert replays[0].prompt_version == "v1" and b.calls == [MSGS]
    assert len(list(read_records(tmp_path / "r.jsonl"))) == 3


def test_errors_are_recorded(tmp_path):
    class Broken(FakeModel):
        def complete(self, messages, profile, schema=None):
            raise ModelError("Engine weg")

    r = Recorder(tmp_path / "r.jsonl").run(Broken(), PROFILES["minimal"], "c", MSGS)
    assert r.error == "Engine weg" and r.completion is None
    assert score_records([r], exact_match({"c": "x"}))[0].score == 0.0


def test_replay_needs_inputs(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl", store_inputs=False)
    rec.run(FakeModel(), PROFILES["minimal"], "c", MSGS)
    with pytest.raises(ValueError, match="Replay"):
        rec.replay(rec.read(), FakeModel(), PROFILES["minimal"])


def test_scorers(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    good = rec.run(FakeModel(answer=lambda m, s: '{"action": "hold"}'), PROFILES["minimal"], "c", MSGS)
    bad = rec.run(FakeModel(answer=lambda m, s: "kein json"), PROFILES["minimal"], "d", MSGS)
    s = json_valid(["action"])
    assert s(good) == 1.0 and s(bad) == 0.0
    assert exact_match({"c": '{"action": "hold"}'})(good) == 1.0
    assert exact_match({})(good) is None


# ------------------------------------------------------------------------------------------------ Statistik


def test_block_bootstrap_wider_than_naive_when_blocks_are_correlated():
    # 10 Blöcke à 10 Werte, innerhalb eines Blocks identisch: effektiv nur 10 unabhängige Werte
    values, blocks = [], []
    for b in range(10):
        values += [float(b % 2)] * 10
        blocks += [b] * 10
    naive = block_bootstrap(values, list(range(100)), seed=1)
    grouped = block_bootstrap(values, blocks, seed=1)
    assert naive.mean == grouped.mean == 0.5
    assert (grouped.high - grouped.low) > 2 * (naive.high - naive.low)
    assert grouped.n == 100 and grouped.n_blocks == 10


def test_paired_bootstrap_detects_consistent_difference():
    a = {i: 1.0 + 0.01 * (i % 3) for i in range(40)}
    b = {i: 0.0 + 0.01 * (i % 5) for i in range(40)}
    est = paired_block_bootstrap(a, b, {i: i // 5 for i in range(40)})
    assert est.excludes_zero and est.mean > 0.9 and est.n_blocks == 8
    same = paired_block_bootstrap(a, dict(a), {i: i // 5 for i in range(40)})
    assert not same.excludes_zero


def test_bootstrap_input_checks():
    with pytest.raises(ValueError):
        block_bootstrap([], [])
    with pytest.raises(ValueError):
        block_bootstrap([1.0], [1, 2])
    with pytest.raises(ValueError):
        paired_block_bootstrap({1: 1.0}, {2: 1.0}, {1: 0, 2: 0})


# --------------------------------------------------------------------------------------------------- Report


def test_report(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    good, bad = FakeModel("gut", answer=lambda m, s: "ja"), FakeModel("schlecht", answer=lambda m, s: "nein")
    expected = {}
    for i in range(16):
        msgs = [{"role": "user", "content": f"Frage {i}"}]
        expected[f"c{i}"] = "ja"
        for m in (good, bad):
            rec.run(m, PROFILES["minimal"], f"c{i}", msgs, tags={"week": i // 2})
    scored = score_records(rec.read(), exact_match(expected))

    def block(r):
        return r.tags["week"]

    summary = summarize(scored, block)
    keys = sorted(summary)
    assert summary[keys[0]]["score"].n_blocks == 8
    diff = compare(
        scored,
        f"gut · minimal ({PROFILES['minimal'].hash})",
        f"schlecht · minimal ({PROFILES['minimal'].hash})",
        block,
    )
    assert diff.mean == 1.0 and diff.excludes_zero
    md = markdown(summary, comparisons={"gut − schlecht": diff})
    assert "| gut · minimal" in md and "Unterschied belastbar" in md and "Opus 5.5 denkt immer" in md
