"""M5: prices, micro-dollar accounting, atomic reservations, paid-link skipping (ADR-0004, PLAN §5)."""

import asyncio
import hashlib
import tempfile
from pathlib import Path

import fakeredis
import pytest
import yaml
from fastapi.testclient import TestClient
from helpers import PROVIDERS_FILE

from llm_gateway.budget.memory import MemoryBudget
from llm_gateway.budget.redis_store import RedisBudget
from llm_gateway.budget.service import BudgetService
from llm_gateway.budget.store import BudgetUnavailable, Caps, Scopes
from llm_gateway.main import create_app
from llm_gateway.pricing.prices import Price, PriceBook, Usage, cost_micro, max_cost_micro, reference_micro
from llm_gateway.providers.mock import MockProvider
from llm_gateway.routing.config import AliasPolicy, Target
from llm_gateway.settings import Settings

HAIKU = Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125)
SCOPES = Scopes("budget:run", "budget:day:d:client:a", "budget:day:d:global")


# --- prices -------------------------------------------------------------------------------------


def test_cost_is_exact_in_micro_dollars():
    # 0.10 USD/Mtok == 0.10 micro-dollars per token
    assert cost_micro(HAIKU, Usage(prompt=1000, completion=200)) == 200
    assert cost_micro(HAIKU, Usage(prompt=1, completion=0)) == 1  # rounded UP, never below real spend


def test_cost_splits_cache_read_and_write():
    usage = Usage(prompt=1000, completion=0, cached=600, cache_creation=200)
    # plain 200*0.10 + read 600*0.01 + write 200*0.125 = 20 + 6 + 25
    assert cost_micro(HAIKU, usage) == 51


def test_usage_reads_openai_convention_and_clamps():
    u = Usage.from_completion(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 150, "cache_creation_tokens": 50},
        }
    )
    assert u is not None and (u.prompt, u.cached, u.cache_creation) == (100, 100, 0)
    assert Usage.from_completion(None) is None


def test_free_tier_costs_zero_but_has_a_reference_price():
    free = Price(reference_input=0.59, reference_output=0.79)
    usage = Usage(prompt=1000, completion=1000)
    assert cost_micro(free, usage) == 0
    assert reference_micro(free, usage) == 1380
    assert reference_micro(HAIKU, usage) == cost_micro(HAIKU, usage)


def test_max_cost_covers_prompt_cache_write_rate():
    plain = max_cost_micro(HAIKU, 1000, 100)
    cached = max_cost_micro(HAIKU, 1000, 100, prompt_cache=True)
    assert plain == 150 and cached == 175


def test_price_book_falls_back_to_default_then_free():
    book = PriceBook({"groq": {"default": Price(reference_input=1.0)}, "anthropic": {"m": HAIKU}})
    assert book.price("anthropic", "m") is HAIKU
    assert book.price("groq", "anything").reference_input == 1.0
    assert not book.price("nobody", "x").paid


# --- stores: same behaviour on memory and on Redis (Lua) ----------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def _fake_redis():
    return fakeredis.FakeAsyncRedis()


@pytest.fixture(params=["memory", "redis"])
async def store(request):
    if request.param == "memory":
        return MemoryBudget()
    return RedisBudget(await _fake_redis())


CAPS_10 = Caps(run=10, client_day=None, global_day=None)


async def test_fifty_concurrent_requests_fit_exactly_ten(store):
    results = await asyncio.gather(*(store.reserve(f"r{i}", 1, 60, SCOPES, CAPS_10) for i in range(50)))
    assert sum(results) == 10
    snap = await store.snapshot(SCOPES)
    assert snap["run"] == {"spent": 0, "reserved": 10}


async def test_three_caps_are_atomic(store):
    caps = Caps(run=100, client_day=5, global_day=100)
    assert await store.reserve("a", 5, 60, SCOPES, caps)
    assert not await store.reserve("b", 1, 60, SCOPES, caps)  # client_day full: nothing reserved elsewhere
    snap = await store.snapshot(SCOPES)
    assert snap["run"]["reserved"] == 5 and snap["global_day"]["reserved"] == 5


async def test_settle_replaces_the_hold_with_the_real_cost(store):
    caps = Caps(run=100, client_day=None, global_day=None)
    await store.reserve("a", 40, 60, SCOPES, caps)
    assert await store.settle("a", 12, SCOPES) is False
    snap = await store.snapshot(SCOPES)
    assert snap["run"] == {"spent": 12, "reserved": 0}
    assert await store.reserve("b", 88, 60, SCOPES, caps)  # room freed by the settle
    assert not await store.reserve("c", 1, 60, SCOPES, caps)


async def test_expired_hold_is_released_and_late_settle_still_counts():
    clock = Clock()
    store = MemoryBudget(clock)
    caps = Caps(run=10, client_day=None, global_day=None)
    assert await store.reserve("a", 10, 30, SCOPES, caps)
    assert not await store.reserve("b", 1, 30, SCOPES, caps)
    clock.now += 31
    assert await store.reserve("b", 10, 30, SCOPES, caps)  # "a" expired, its hold is gone
    assert await store.settle("a", 4, SCOPES) is True  # late
    snap = await store.snapshot(SCOPES)
    assert snap["run"] == {"spent": 4, "reserved": 10}


async def test_redis_expired_hold_is_released_and_late_settle_counts():
    redis = await _fake_redis()
    store = RedisBudget(redis)
    caps = Caps(run=10, client_day=None, global_day=None)
    assert await store.reserve("a", 10, 0, SCOPES, caps)  # ttl 0 -> int(0)+1 = 1 s
    await asyncio.sleep(2.1)
    assert await store.reserve("b", 10, 30, SCOPES, caps)
    assert await store.settle("a", 4, SCOPES) is True
    snap = await store.snapshot(SCOPES)
    assert snap["run"] == {"spent": 4, "reserved": 10}


async def test_budget_keys_have_no_ttl():
    redis = await _fake_redis()
    store = RedisBudget(redis)
    await store.reserve("a", 5, 60, SCOPES, CAPS_10)
    await store.settle("a", 3, SCOPES)
    for key in (*SCOPES.as_list(), "budget:reservations", "budget:reservation_meta"):
        assert await redis.pttl(key) in (-1, -2), key  # -1: key without TTL; -2: gone. Never a TTL.
    assert await redis.pttl(SCOPES.run) == -1  # volatile-lfu cannot evict it


async def test_redis_down_raises_budget_unavailable():
    class Down:
        def register_script(self, _):
            async def boom(**_kw):
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("down")

            return boom

    store = RedisBudget(Down())
    with pytest.raises(BudgetUnavailable):
        await store.reserve("a", 1, 60, SCOPES, CAPS_10)
    with pytest.raises(BudgetUnavailable):
        await store.settle("a", 1, SCOPES)


# --- service / gate -----------------------------------------------------------------------------


def _service(store=None, run_cap=1.0, **kw):
    prices = PriceBook({"anthropic": {"claude-haiku-5-5": HAIKU}})
    return BudgetService(store or MemoryBudget(), prices, {"anthropic"}, run_cap_usd=run_cap, **kw)


def _policy(**kw):
    return AliasPolicy(chain=[Target(provider="anthropic", model="claude-haiku-5-5")], **kw)


async def test_gate_reserves_worst_case_then_settles_real_cost():
    seen = []
    svc = _service(on_cost=lambda *a: seen.append(a))
    gate = svc.gate("a", 1.0, _policy(), 1000, 100, False)
    target = Target(provider="anthropic", model="claude-haiku-5-5")
    res = await gate.reserve(target)
    assert res.amount == 150  # 1000*0.10 + 100*0.50
    snap = await svc.snapshot("a", 1.0)
    assert snap["scopes"]["run"]["reserved"] == 150
    cost, ref = await gate.settle(res, {"prompt_tokens": 1000, "completion_tokens": 20})
    assert (cost, ref) == (110, 110) and seen == [("a", "anthropic", 110, 110)]
    snap = await svc.snapshot("a", 1.0)
    assert snap["scopes"]["run"]["spent"] == 110 and snap["scopes"]["run"]["reserved"] == 0


async def test_settle_without_usage_charges_input_plus_emitted():
    svc = _service()
    gate = svc.gate("a", 1.0, _policy(), 1000, 100, False)
    res = await gate.reserve(Target(provider="anthropic", model="claude-haiku-5-5"))
    cost, _ = await gate.settle(res, None, emitted_tokens=10)
    assert cost == 105  # 1000*0.10 + 10*0.50
    assert svc.counters["usage_estimated"] == 1


async def test_failed_call_charges_nothing():
    svc = _service()
    gate = svc.gate("a", 1.0, _policy(), 1000, 100, False)
    res = await gate.reserve(Target(provider="anthropic", model="claude-haiku-5-5"))
    assert await gate.settle(res, failed=True) == (0, 0)
    snap = await svc.snapshot("a", 1.0)
    assert snap["scopes"]["run"] == {"spent": 0, "reserved": 0, "cap": 1_000_000, "remaining": 1_000_000}


async def test_client_cap_zero_blocks_paid():
    from llm_gateway.budget.service import BudgetSkip

    svc = _service()
    gate = svc.gate("a", 0.0, _policy(), 10, 10, False)
    with pytest.raises(BudgetSkip) as exc:
        await gate.reserve(Target(provider="anthropic", model="claude-haiku-5-5"))
    assert exc.value.reason == "cap" and gate.skipped == ["anthropic"]


async def test_store_down_fails_closed_for_paid():
    from llm_gateway.budget.service import BudgetSkip

    class Down:
        async def reserve(self, *a, **k):
            raise BudgetUnavailable("down")

    svc = _service(store=Down())
    gate = svc.gate("a", 1.0, _policy(), 10, 10, False)
    with pytest.raises(BudgetSkip) as exc:
        await gate.reserve(Target(provider="anthropic", model="claude-haiku-5-5"))
    assert exc.value.reason == "unavailable" and svc.counters["unavailable"] == 1


# --- end to end ---------------------------------------------------------------------------------

RICH_KEY, POOR_KEY = "gw_rich_" + "r" * 32, "gw_poor_" + "p" * 32
_DIR = Path(tempfile.mkdtemp(prefix="gw-budget-"))


class PaidMock(MockProvider):
    """Stands in for Haiku: counts calls and reports a fixed usage (1000 in, 200 out -> 200 micro-dollars)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def complete(self, request, model):
        self.calls += 1
        out = await super().complete(request, model)
        out["usage"] = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
        return out

    async def stream(self, request, model):
        self.calls += 1
        async for chunk in super().stream(request, model):
            if "usage" in chunk:
                chunk["usage"] = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
            yield chunk


def _app(run_budget_usd=1.0, **overrides):
    routes = {
        "aliases": {
            "paid": {
                "chain": [
                    {"provider": "anthropic", "model": "claude-haiku-5-5"},
                    {"provider": "mock", "model": "mock-x"},
                ],
                "max_tokens": 100,
                "max_input_tokens": 50,
            },
            "onlypaid": {
                "chain": [{"provider": "anthropic", "model": "claude-haiku-5-5"}],
                "max_tokens": 100,
            },
        }
    }
    clients = {
        "clients": {
            "rich": {
                "key_sha256": hashlib.sha256(RICH_KEY.encode()).hexdigest(),
                "allowed_aliases": ["paid", "onlypaid"],
                "budget_usd_per_day": 1.0,
            },
            "poor": {
                "key_sha256": hashlib.sha256(POOR_KEY.encode()).hexdigest(),
                "allowed_aliases": ["paid", "onlypaid"],
            },
        }
    }
    routes_file, clients_file = _DIR / "routes.yaml", _DIR / "clients.yaml"
    routes_file.write_text(yaml.safe_dump(routes), encoding="utf-8")
    clients_file.write_text(yaml.safe_dump(clients), encoding="utf-8")
    settings = Settings(
        _env_file=None,
        env="test",
        routes_file=routes_file,
        clients_file=clients_file,
        providers_file=PROVIDERS_FILE,
        groq_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
        cache_backend="off",
        embedder="off",
        run_budget_usd=run_budget_usd,
        **overrides,
    )
    app = create_app(settings)
    paid = PaidMock()
    app.state.gateway.providers["anthropic"] = paid
    return app, paid


def _post(client, key, alias, content="hello", **extra):
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": alias, "messages": [{"role": "user", "content": content}], **extra},
    )


def test_paid_call_is_charged_and_visible_in_admin_budget():
    app, paid = _app()
    with TestClient(app) as c:
        r = _post(c, RICH_KEY, "paid")
        assert r.status_code == 200 and paid.calls == 1
        assert "x-budget-skipped" not in r.headers
        b = c.get("/admin/budget", headers={"Authorization": f"Bearer {RICH_KEY}"}).json()
    assert b["scopes"]["run"]["spent"] == 200 and b["scopes"]["run"]["reserved"] == 0
    assert b["scopes"]["client_day"]["cap"] == 1_000_000
    assert [(r["provider"], r["actual"]) for r in b["cost_micro_usd"]] == [("anthropic", 200)]


def test_client_without_budget_skips_paid_and_falls_back():
    app, paid = _app()
    with TestClient(app) as c:
        r = _post(c, POOR_KEY, "paid")
    assert r.status_code == 200 and paid.calls == 0
    assert r.headers["x-budget-skipped"] == "anthropic"
    assert r.headers["x-gateway-provider"].startswith("mock")


def test_only_paid_links_left_returns_402():
    app, paid = _app()
    with TestClient(app) as c:
        r = _post(c, POOR_KEY, "onlypaid")
    assert r.status_code == 402 and paid.calls == 0
    err = r.json()["error"]
    assert err["code"] == "budget_exceeded" and err["type"] == "insufficient_quota"


def test_run_cap_zero_disables_paid_providers():
    app, paid = _app(run_budget_usd=0.0)
    with TestClient(app) as c:
        r = _post(c, RICH_KEY, "paid")
    assert r.status_code == 200 and paid.calls == 0 and r.headers["x-budget-skipped"] == "anthropic"


def test_cap_runs_out_mid_run():
    # Worst case per call is ~50 micro-dollars (100 max_tokens * 0.5); each settles at 200. Cap 500: after
    # two settled calls (400) the 100 left still fits a hold; after three (600) it does not.
    app, paid = _app(run_budget_usd=0.0005)
    with TestClient(app) as c:
        codes = [_post(c, RICH_KEY, "paid").headers.get("x-budget-skipped") for _ in range(4)]
    assert codes[:2] == [None, None] and codes[-1] == "anthropic"
    assert paid.calls < 4


def test_input_over_alias_limit_is_400():
    app, _ = _app()
    with TestClient(app) as c:
        r = _post(c, RICH_KEY, "paid", content="word " * 400)
    assert r.status_code == 400 and r.json()["error"]["code"] == "input_too_long"


def test_stream_settles_with_usage():
    app, paid = _app()
    with TestClient(app) as c:
        r = _post(c, RICH_KEY, "paid", stream=True)
        assert r.status_code == 200 and "[DONE]" in r.text
        b = c.get("/admin/budget", headers={"Authorization": f"Bearer {RICH_KEY}"}).json()
    assert paid.calls == 1
    assert b["scopes"]["run"]["spent"] == 200 and b["scopes"]["run"]["reserved"] == 0


def test_free_provider_cost_is_zero_with_reference_price_in_stats():
    app, _ = _app()
    with TestClient(app) as c:
        _post(c, POOR_KEY, "paid")
        s = c.get("/stats").json()
    assert all(row["actual"] == 0 for row in s["cost_micro_usd"])
