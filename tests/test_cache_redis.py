"""Redis cache backend, on fakeredis (no server needed). The same contract runs against MemoryBackend, so the two
cannot drift. NOT covered: a real Redis, `volatile-lfu` eviction and the Query Engine (HNSW); see the backend's
docstring."""

import fakeredis
import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from helpers import FULL_KEY, make_app
from redis.exceptions import ConnectionError as RedisConnectionError
from test_cache import PASSWORD, PASSWORD_PARAPHRASE, POLICY, body, completion, lookup

from llm_gateway.cache.backend import CacheEntry
from llm_gateway.cache.config import CacheConfig, Thresholds
from llm_gateway.cache.embedder import HashEmbedder
from llm_gateway.cache.memory import MemoryBackend
from llm_gateway.cache.redis_backend import RedisBackend, pack, unpack
from llm_gateway.cache.service import CacheService

NOW = 1_000_000.0


class Clock:
    now = NOW

    def __call__(self):
        return self.now


def entry(key="k1", ns="ns", client="c", tags=("t",), ttl=3600, text="hello", clock=NOW):
    return CacheEntry(
        key, ns, client, "fast", completion("hi"), "groq:m", text, tuple(tags), clock, clock + ttl
    )


@pytest.fixture(params=["memory", "redis"])
def backend(request):
    clock = Clock()
    if request.param == "memory":
        b = MemoryBackend(clock=clock)
    else:
        b = RedisBackend(FakeAsyncRedis(), clock=clock)
    b.clock = clock
    return b


# --- the contract both backends keep --------------------------------------------------------------


async def test_put_then_exact_get_returns_the_entry(backend):
    await backend.put(entry(), None)
    got = await backend.get_exact("k1")
    assert got.response == completion("hi") and got.provider == "groq:m" and got.tags == ("t",)
    assert await backend.get_exact("nope") is None


async def test_entries_expire_at_their_deadline(backend):
    await backend.put(entry(ttl=10), None)
    backend.clock.now = NOW + 11
    assert await backend.get_exact("k1") is None


async def test_search_ranks_by_cosine_and_stays_in_its_namespace(backend):
    await backend.put(entry("a", ns="n1"), [1.0, 0.0])
    await backend.put(entry("b", ns="n1"), [0.6, 0.8])
    await backend.put(entry("c", ns="n2"), [1.0, 0.0])
    found = await backend.search("n1", [1.0, 0.0], k=5)
    assert [c.entry.key for c in found] == ["a", "b"]
    assert found[0].similarity == pytest.approx(1.0) and found[1].similarity == pytest.approx(0.6)
    assert [c.entry.key for c in await backend.search("n1", [1.0, 0.0], k=1)] == ["a"]


async def test_search_skips_expired_entries(backend):
    await backend.put(entry("a", ns="n", ttl=5), [1.0, 0.0])
    await backend.put(entry("b", ns="n", ttl=500), [0.0, 1.0])
    backend.clock.now = NOW + 6
    assert [c.entry.key for c in await backend.search("n", [1.0, 0.0])] == ["b"]


async def test_touch_counts_hits(backend):
    await backend.put(entry(), None)
    await backend.touch("k1")
    await backend.touch("k1")
    assert (await backend.get_exact("k1")).hits == 2


async def test_rewriting_a_key_replaces_its_tags_and_vector(backend):
    await backend.put(entry(tags=("old",), ns="n"), [1.0, 0.0])
    await backend.put(entry(tags=("new",), ns="n"), [0.0, 1.0])
    assert await backend.invalidate({"old"}) == 0
    assert [c.similarity for c in await backend.search("n", [0.0, 1.0])] == [pytest.approx(1.0)]
    assert await backend.invalidate({"new"}) == 1
    assert await backend.get_exact("k1") is None


async def test_invalidate_by_tag_can_be_limited_to_one_client(backend):
    await backend.put(entry("a", client="x", tags=("faq",)), None)
    await backend.put(entry("b", client="y", tags=("faq",)), None)
    assert await backend.invalidate({"faq"}, client="x") == 1
    assert await backend.get_exact("a") is None and await backend.get_exact("b") is not None
    assert await backend.invalidate({"faq"}) == 1


async def test_invalidated_entries_leave_the_semantic_index(backend):
    await backend.put(entry("a", ns="n", tags=("t",)), [1.0, 0.0])
    await backend.invalidate({"t"})
    assert await backend.search("n", [1.0, 0.0]) == []


# --- what only Redis does -------------------------------------------------------------------------


async def test_entries_survive_a_new_backend_on_the_same_server():
    server = fakeredis.FakeServer()
    first = RedisBackend(FakeAsyncRedis(server=server), clock=Clock())
    await first.put(entry(ns="n"), [1.0, 0.0])
    second = RedisBackend(FakeAsyncRedis(server=server), clock=Clock())  # "another worker" or a restart
    assert (await second.get_exact("k1")).response == completion("hi")
    assert [c.entry.key for c in await second.search("n", [1.0, 0.0])] == ["k1"]


async def test_redis_ttl_is_set_so_eviction_policy_can_reclaim_entries():
    redis = FakeAsyncRedis()
    await RedisBackend(redis, clock=Clock()).put(entry(ttl=100), None)
    assert 0 < await redis.ttl("gwc:e:k1") <= 100


async def test_a_vector_row_whose_entry_was_evicted_is_cleaned_up_on_search():
    redis = FakeAsyncRedis()
    b = RedisBackend(redis, clock=Clock())
    await b.put(entry("a", ns="n"), [1.0, 0.0])
    await redis.delete("gwc:e:a")  # what `volatile-lfu` does under memory pressure
    assert await b.search("n", [1.0, 0.0]) == []
    assert await redis.hlen("gwc:ns:n") == 0


def test_vectors_round_trip_through_float32():
    v = [0.25, -0.5, 0.125]
    assert unpack(pack(v)) == v


class Broken(FakeAsyncRedis):
    async def execute_command(self, *a, **kw):
        raise RedisConnectionError("down")


async def test_a_redis_outage_degrades_reads_and_writes_to_misses_and_counts_errors():
    b = RedisBackend(Broken(), clock=Clock())
    await b.put(entry(), [1.0])
    assert await b.get_exact("k1") is None
    assert await b.search("n", [1.0]) == []
    await b.touch("k1")
    assert b.stats()["errors"] >= 4


async def test_an_outage_during_invalidation_is_raised_not_hidden():
    with pytest.raises(RedisConnectionError):
        await RedisBackend(Broken(), clock=Clock()).invalidate({"t"})


# --- through the cache service and the app --------------------------------------------------------


async def test_the_service_serves_exact_and_semantic_hits_from_redis():
    config = CacheConfig(embedders={"hash": Thresholds(t_high=0.8, t_low=0.7)})
    svc = CacheService(RedisBackend(FakeAsyncRedis()), config, HashEmbedder())
    req = svc.prepare("c", "fast", POLICY, body(PASSWORD), {})
    assert svc.store_later(req, completion("answer"), "groq:m", degraded=False)
    await svc.flush()
    assert (await lookup(svc, PASSWORD)).layer == "exact"
    sem = await lookup(svc, PASSWORD_PARAPHRASE)
    assert sem.layer == "semantic" and sem.entry.response == completion("answer")
    assert await lookup(svc, "How do I change my email address?") is None
    assert svc.snapshot()["backend"] == "redis"


def test_the_app_builds_the_redis_backend_and_a_repeat_is_served_from_it(monkeypatch):
    shared = FakeAsyncRedis()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *_a, **_k: shared)
    app = make_app(cache_backend="redis", embedder="off")
    gw = app.state.gateway
    gw.routes.aliases["fast"].chain[-1].degraded = False  # only `mock` answers in tests
    client = TestClient(app, headers={"Authorization": f"Bearer {FULL_KEY}"})
    ask = {"model": "fast", "messages": [{"role": "user", "content": "hola redis"}]}
    first = client.post("/v1/chat/completions", json=ask)
    assert first.status_code == 200 and "x-cache-layer" not in first.headers
    import asyncio

    asyncio.run(gw.cache.flush())
    second = client.post("/v1/chat/completions", json=ask)
    assert second.headers.get("x-cache-layer") == "exact"
    assert any(k.startswith(b"gwc:e:") for k in asyncio.run(shared.keys("gwc:e:*")))
