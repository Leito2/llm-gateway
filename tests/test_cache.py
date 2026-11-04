"""M4 · Cache: keys, guards, backend, service logic and the request path (R07, R08, R09, R21).

Everything runs offline with `HashEmbedder` (spelling overlap, no model) and an in-memory backend. The real
embedders are measured separately (G6, G9, G11)."""

import asyncio
import base64
import json
import struct

import pytest
from fastapi.testclient import TestClient
from helpers import DEMO_KEY, FULL_KEY, make_app, make_client

from llm_gateway.api.schemas import ChatCompletionRequest, ChatMessage
from llm_gateway.cache import guards
from llm_gateway.cache.backend import CacheEntry
from llm_gateway.cache.config import CacheConfig, Thresholds
from llm_gateway.cache.embedder import HashEmbedder, cosine
from llm_gateway.cache.keys import (
    exact_key,
    has_non_text,
    namespace_base,
    parse_tags,
    semantic_namespace,
    semantic_text,
    system_hash,
)
from llm_gateway.cache.memory import MemoryBackend
from llm_gateway.cache.service import CacheService
from llm_gateway.providers.mock import MockProvider
from llm_gateway.routing.config import AliasPolicy

URL = "/v1/chat/completions"
PASSWORD = "How do I reset my password?"
PASSWORD_PARAPHRASE = "how can I reset my password"


def msgs(*pairs):
    return [ChatMessage(role=r, content=c) for r, c in pairs]


# --- keys -----------------------------------------------------------------------------------------------


def test_normalization_ignores_spacing_but_not_case():
    a = msgs(("user", "hola   mundo\n"))
    b = msgs(("user", "hola mundo"))
    base = namespace_base("c", "fast", "s", "", None)
    assert exact_key(base, a) == exact_key(base, b)
    assert exact_key(base, msgs(("user", "US"))) != exact_key(base, msgs(("user", "us")))


def test_namespace_separates_clients_aliases_system_prompts_versions_formats_and_embedders():
    sysa = system_hash(msgs(("system", "A"), ("user", "q")))
    sysb = system_hash(msgs(("system", "B"), ("user", "q")))
    variants = {
        namespace_base("c", "fast", sysa, "", None),
        namespace_base("d", "fast", sysa, "", None),
        namespace_base("c", "smart", sysa, "", None),
        namespace_base("c", "fast", sysb, "", None),
        namespace_base("c", "fast", sysa, "v2", None),
        namespace_base("c", "fast", sysa, "", {"type": "json_object"}),
    }
    assert len(variants) == 6
    base = namespace_base("c", "fast", sysa, "", None)
    assert semantic_namespace(base, "e5-small") != semantic_namespace(base, "nomic")


def test_the_whole_conversation_and_its_roles_are_in_the_exact_key():
    base = namespace_base("c", "fast", "s", "", None)
    one = msgs(("user", "a"), ("assistant", "b"), ("user", "c"))
    two = msgs(("user", "x"), ("assistant", "b"), ("user", "c"))
    swapped = msgs(("assistant", "a"), ("user", "b"), ("user", "c"))
    assert len({exact_key(base, one), exact_key(base, two), exact_key(base, swapped)}) == 3


def test_semantic_text_rules():
    single = msgs(("system", "s"), ("user", " What  is X? "))
    assert semantic_text(single, "single") == "What is X?"
    multi = msgs(("user", "a"), ("assistant", "b"), ("user", "c"))
    assert semantic_text(multi, "single") is None  # earlier turns could change the meaning
    assert semantic_text(multi, "last_turn") == "c"  # FAQ style: each question stands alone
    assert semantic_text(msgs(("system", "s")), "last_turn") is None


def test_requests_with_images_are_never_cached():
    image = ChatMessage(role="user", content=[{"type": "text", "text": "hi"}, {"type": "image_url"}])
    assert has_non_text([image]) and not has_non_text(msgs(("user", "hi")))


def test_tags_are_parsed_deduplicated_and_unsafe_ones_dropped():
    assert parse_tags("doc:1, doc:7,doc:1, bad tag, {x}") == ("doc:1", "doc:7")
    assert parse_tags(None) == () and len(parse_tags(",".join(f"t{i}" for i in range(50)))) == 20


# --- lexical guards (R08) ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("new", "cached", "guard"),
    [
        ("Cancel my order of $50", "Cancel my order of $500", "numbers"),
        ("Pagar 1.500 pesos", "Pagar 15.00 pesos", "numbers"),
        ("Cuánto cuesta 100 USD", "Cuánto cuesta 100 EUR", "currency"),
        ("Reunión el lunes", "Reunión el martes", "dates"),
        ("Where is part SKU-A1?", "Where is part SKU-B1?", "codes"),
        ("No me llegó el pedido", "Me llegó el pedido", "negation"),
        ("I can't log in", "I can log in", "negation"),
        ("It doesn't work", "It works", "negation"),
        ("no, no quiero", "no quiero", "negation"),
    ],
)
def test_r08_guards_reject_pairs_that_differ_in_a_fact(new, cached, guard):
    assert guards.mismatch(new, cached) == guard


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("How do I reset my password?", "how can I reset my password"),
        ("Pay 1,000 pesos", "Pay 1000 pesos"),
        ("Quiero cambiar mi contraseña", "Quiero cambiar la contraseña"),
        ("I can't log in", "I cannot log in"),
    ],
)
def test_guards_let_harmless_variants_through(new, cached):
    assert guards.mismatch(new, cached) is None


# --- thresholds, embedder, backend ------------------------------------------------------------------------


def test_thresholds_validation():
    assert not Thresholds().calibrated
    assert Thresholds(t_high=0.9).t_low == 0.9  # no gray zone
    with pytest.raises(ValueError):
        Thresholds(t_high=0.8, t_low=0.9)
    with pytest.raises(ValueError):
        Thresholds(t_low=0.5)


async def test_hash_embedder_is_normalised_deterministic_and_prefers_similar_text():
    e = HashEmbedder()
    a, b, c = await e.embed([PASSWORD, PASSWORD_PARAPHRASE, "Where is the nearest store?"])
    assert abs(cosine(a, a) - 1) < 1e-9 and (await e.embed([PASSWORD]))[0] == a
    assert cosine(a, b) > 0.8 > cosine(a, c)


def entry(key="k", ns="n", client="c", tags=(), now=100.0, ttl=60.0, hits=0):
    return CacheEntry(key, ns, client, "fast", {"x": [1]}, "groq:m", "q", tags, now, now + ttl, hits)


async def test_backend_ttl_search_order_and_isolation():
    clock = [100.0]
    be = MemoryBackend(clock=lambda: clock[0])
    await be.put(entry("a"), [1.0, 0.0])
    await be.put(entry("b"), [0.6, 0.8])
    await be.put(entry("o", ns="other"), [1.0, 0.0])
    found = await be.search("n", [1.0, 0.0])
    assert [c.entry.key for c in found] == ["a", "b"] and found[0].similarity == 1.0
    got = await be.get_exact("a")
    got.response["x"].append(2)  # a caller cannot corrupt the stored copy
    assert (await be.get_exact("a")).response == {"x": [1]}
    clock[0] = 160.0  # TTL: logical expiry, entry is gone
    assert await be.get_exact("a") is None and await be.search("n", [1.0, 0.0]) == []


async def test_r09_backend_invalidates_by_tag_and_only_for_the_given_client():
    be = MemoryBackend()
    now = 10**10
    await be.put(entry("a", client="c", tags=("doc:1", "doc:2"), now=now), [1.0])
    await be.put(entry("b", client="d", tags=("doc:1",), now=now), [1.0])
    await be.put(entry("c", client="c", tags=("doc:9",), now=now), [1.0])
    assert await be.invalidate({"doc:1"}, client="c") == 1
    assert await be.get_exact("a") is None and await be.get_exact("b") and await be.get_exact("c")
    assert await be.invalidate({"doc:1"}) == 1 and await be.get_exact("b") is None


async def test_backend_evicts_the_least_hit_entry_when_full_and_rewrites_replace_tags():
    be = MemoryBackend(max_entries=2)
    now = 10**10
    await be.put(entry("hot", hits=5, now=now), [1.0])
    await be.put(entry("cold", hits=0, now=now + 1), [1.0])
    await be.put(entry("new", now=now + 2), [1.0])
    assert await be.get_exact("cold") is None and await be.get_exact("hot") and be.evicted == 1
    await be.put(entry("hot", tags=("t",), now=now), [1.0])
    await be.put(entry("hot", tags=("u",), now=now), [1.0])
    assert await be.invalidate({"t"}) == 0 and await be.invalidate({"u"}) == 1


# --- service logic ----------------------------------------------------------------------------------------


class FakeVerifier:
    def __init__(self, answer=True, boom=False):
        self.answer, self.boom, self.calls = answer, boom, 0

    async def equivalent(self, new, cached):
        self.calls += 1
        if self.boom:
            raise RuntimeError("model crashed")
        return self.answer


class SlowEmbedder(HashEmbedder):
    async def embed(self, texts):
        await asyncio.sleep(5)
        return await super().embed(texts)


POLICY = AliasPolicy(chain=[{"provider": "mock", "model": "m"}])


def body(text, **kw):
    return ChatCompletionRequest(model="fast", messages=[{"role": "user", "content": text}], **kw)


def completion(text="answer", finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def service(t_high=0.8, t_low=0.7, verifier=None, embedder=None, **cfg):
    config = CacheConfig(embedders={"hash": Thresholds(t_high=t_high, t_low=t_low)}, **cfg)
    return CacheService(MemoryBackend(), config, embedder or HashEmbedder(), verifier)


async def put(svc, text, answer="answer", **kw):
    req = svc.prepare("c", "fast", POLICY, body(text), kw.get("headers", {}))
    assert svc.store_later(req, completion(answer), "groq:m", degraded=False)
    await svc.flush()


async def lookup(svc, text, headers=None):
    return await svc.lookup(svc.prepare("c", "fast", POLICY, body(text), headers or {}))


async def test_exact_hit_beats_semantic_and_semantic_hit_needs_the_threshold():
    svc = service()
    await put(svc, PASSWORD)
    exact = await lookup(svc, PASSWORD)
    sem = await lookup(svc, PASSWORD_PARAPHRASE)
    far = await lookup(svc, "How do I change my email address?")
    assert exact.layer == "exact" and exact.similarity is None
    assert sem.layer == "semantic" and sem.similarity > 0.8
    assert far is None and svc.counters["miss"] == 1


async def test_r08_a_close_embedding_is_still_a_miss_when_a_guard_fails():
    svc = service()
    await put(svc, "reset password in 5 minutes")
    assert await lookup(svc, "reset password in 50 minutes") is None  # similarity 0.91, numbers differ
    assert svc.guard_rejections["numbers"] == 1


async def test_gray_zone_goes_to_the_verifier_and_its_no_or_failure_is_a_miss():
    for verifier, expected in [
        (FakeVerifier(True), "verified"),
        (FakeVerifier(False), None),
        (FakeVerifier(boom=True), None),
        (None, None),
    ]:
        svc = service(verifier=verifier)
        await put(svc, "What is your refund policy?")
        hit = await lookup(svc, "What's the refund policy")  # similarity 0.74: gray zone
        assert (hit.layer if hit else None) == expected


async def test_the_verifier_is_not_asked_above_t_high_or_below_t_low():
    verifier = FakeVerifier()
    svc = service(verifier=verifier)
    await put(svc, PASSWORD)
    await lookup(svc, PASSWORD_PARAPHRASE)  # 0.85: no verifier needed
    await lookup(svc, "Where is the nearest store?")  # 0.0: below t_low
    assert verifier.calls == 0


async def test_uncalibrated_embedder_observes_and_never_serves():
    svc = service(t_high=None, t_low=None)
    await put(svc, PASSWORD)
    assert await lookup(svc, PASSWORD_PARAPHRASE) is None
    snap = svc.snapshot()
    assert snap["semantic"].startswith("observing") and snap["counters"]["observed"] == 1
    assert snap["recent_observed"][0]["similarity"] > 0.8 and snap["recent_observed"][0]["guard"] is None
    assert (await lookup(svc, PASSWORD)).layer == "exact"  # the exact layer works from day one


async def test_a_slow_embedder_degrades_to_exact_only():
    svc = service(embedder=SlowEmbedder(), embed_timeout_s=0.05)
    req = svc.prepare("c", "fast", POLICY, body(PASSWORD), {})
    assert await svc.lookup(req) is None and svc.counters["semantic_error"] == 1


async def test_without_an_embedder_the_semantic_layer_is_off_but_exact_works():
    svc = CacheService(MemoryBackend(), CacheConfig(), None)
    await put(svc, PASSWORD)
    assert (await lookup(svc, PASSWORD)).layer == "exact" and await lookup(svc, PASSWORD_PARAPHRASE) is None
    assert svc.snapshot()["semantic"].startswith("off")


@pytest.mark.parametrize(
    ("answer", "degraded"),
    [
        (completion(finish="length"), False),
        (completion(finish="content_filter"), False),
        (completion(text=""), False),
        (completion(), True),
    ],
)
async def test_unsafe_answers_are_never_stored(answer, degraded):
    svc = service()
    req = svc.prepare("c", "fast", POLICY, body(PASSWORD), {})
    assert not svc.store_later(req, answer, "groq:m", degraded=degraded)
    await svc.flush()
    assert await lookup(svc, PASSWORD) is None and svc.counters["stored"] == 0


async def test_a_full_write_queue_drops_instead_of_waiting():
    svc = service(write_queue=2, embedder=SlowEmbedder(), embed_timeout_s=0.01)
    reqs = [svc.prepare("c", "fast", POLICY, body(f"question {i}"), {}) for i in range(4)]
    queued = [svc.store_later(r, completion(), "groq:m", degraded=False) for r in reqs]
    assert queued == [True, True, False, False] and svc.counters["store_dropped"] == 2
    await svc.flush()


def test_prepare_returns_none_for_opt_outs_and_private_scope():
    svc = service()
    assert svc.prepare("c", "fast", POLICY, body(PASSWORD, cache=False), {}) is None  # opt out
    assert svc.prepare("c", "fast", POLICY, body(PASSWORD), {"x-cache-scope": "private"}) is None  # R21
    off = AliasPolicy(chain=POLICY.chain, cache=False, semantic_cache=False)
    assert svc.prepare("c", "fast", off, body(PASSWORD), {}) is None
    assert svc.prepare("c", "fast", off, body(PASSWORD, cache=True), {}) is not None  # client opt in
    exact_only = AliasPolicy(chain=POLICY.chain, cache=True, semantic_cache=False)
    assert svc.prepare("c", "fast", exact_only, body(PASSWORD), {}).sem_text is None


# --- the request path -------------------------------------------------------------------------------------


class Counting(MockProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    async def complete(self, request, model):
        self.calls += 1
        return await super().complete(request, model)

    async def stream(self, request, model):
        self.calls += 1
        async for chunk in super().stream(request, model):
            yield chunk


def cached_app(t_high=0.8, t_low=0.7, **providers):
    app = make_app(cache_backend="memory", embedder="hash")
    gw = app.state.gateway
    gw.cache.config.embedders["hash"] = Thresholds(t_high=t_high, t_low=t_low) if t_high else Thresholds()
    gw.providers.update(providers)
    return app, gw


def ask(client, text, alias="fast", **extra):
    headers = extra.pop("headers", None)
    return client.post(
        URL, json={"model": alias, "messages": [{"role": "user", "content": text}], **extra}, headers=headers
    )


def flushed(client, gw):
    """Writes are fire-and-forget tasks on the app's loop; extra requests on the same portal let them run."""
    for _ in range(50):
        client.get("/healthz")
        if not gw.cache._pending:
            return
    raise AssertionError("write-back never finished")


def test_r07_a_repeated_question_is_served_from_cache_without_calling_the_provider():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        first = ask(client, PASSWORD)
        flushed(client, gw)
        second = ask(client, PASSWORD)
        assert "x-cache-layer" not in first.headers and first.headers["x-gateway-provider"].startswith(
            "groq:"
        )
        assert (
            second.headers["x-cache-layer"] == "exact" and second.headers["x-gateway-fallback-depth"] == "0"
        )
        assert second.headers["x-gateway-provider"] == first.headers["x-gateway-provider"]
        assert second.json()["choices"] == first.json()["choices"] and groq.calls == 1


def test_r07_a_paraphrase_is_a_semantic_hit_with_its_similarity():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD)
        flushed(client, gw)
        r = ask(client, PASSWORD_PARAPHRASE)
        assert r.headers["x-cache-layer"] == "semantic" and float(r.headers["x-cache-similarity"]) > 0.8
        assert groq.calls == 1
        assert client.get("/stats").json()["cache"]["counters"]["hit_semantic"] == 1


def test_r08_in_the_request_path_a_changed_number_is_a_miss():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, "reset password in 5 minutes")
        flushed(client, gw)
        r = ask(client, "reset password in 50 minutes")
        assert "x-cache-layer" not in r.headers and groq.calls == 2


def test_uncalibrated_semantic_layer_does_not_serve_in_the_request_path():
    groq = Counting()
    app, gw = cached_app(t_high=None, groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD)
        flushed(client, gw)
        assert "x-cache-layer" not in ask(client, PASSWORD_PARAPHRASE).headers
        assert ask(client, PASSWORD).headers["x-cache-layer"] == "exact"
        assert client.get("/stats").json()["cache"]["counters"]["observed"] == 1


def test_r21_private_scope_neither_reads_nor_writes_the_cache():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD, headers={"X-Cache-Scope": "private"})
        flushed(client, gw)
        assert gw.cache.counters["stored"] == 0
        ask(client, PASSWORD)  # a normal request stores...
        flushed(client, gw)
        private = ask(client, PASSWORD, headers={"X-Cache-Scope": "private"})
        assert "x-cache-layer" not in private.headers and groq.calls == 3  # ...but private never reads it


def test_r09_invalidating_a_tag_stops_serving_every_entry_with_it():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD, headers={"X-Cache-Tags": "doc:1,doc:2"})
        ask(client, "What is your refund policy?", headers={"X-Cache-Tags": "doc:9"})
        flushed(client, gw)
        assert ask(client, PASSWORD).headers["x-cache-layer"] == "exact"
        r = client.post("/admin/cache/invalidate", json={"tags": ["doc:2"]})
        assert r.status_code == 200 and r.json() == {"removed": 1, "tags": ["doc:2"]}
        assert "x-cache-layer" not in ask(client, PASSWORD).headers
        assert ask(client, "What is your refund policy?").headers["x-cache-layer"] == "exact"


def test_invalidation_only_reaches_the_callers_own_entries():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        demo = {"Authorization": f"Bearer {DEMO_KEY}"}
        ask(client, PASSWORD, headers={**demo, "X-Cache-Tags": "doc:1"})
        flushed(client, gw)
        assert client.post("/admin/cache/invalidate", json={"tags": ["doc:1"]}).json()["removed"] == 0
        assert ask(client, PASSWORD, headers=demo).headers["x-cache-layer"] == "exact"
        assert client.post("/admin/cache/invalidate", json={"tags": ["bad tag"]}).status_code == 400
        assert TestClient(app).post("/admin/cache/invalidate", json={"tags": ["doc:1"]}).status_code == 401


def test_clients_and_system_prompts_never_share_answers():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        demo = {"Authorization": f"Bearer {DEMO_KEY}"}
        ask(client, PASSWORD)
        flushed(client, gw)
        assert "x-cache-layer" not in ask(client, PASSWORD, headers=demo).headers
        other = {
            "model": "fast",
            "messages": [{"role": "system", "content": "Be rude."}, {"role": "user", "content": PASSWORD}],
        }
        assert "x-cache-layer" not in client.post(URL, json=other).headers
        assert groq.calls == 3


def test_degraded_truncated_and_opted_out_answers_are_not_cached():
    app, gw = cached_app()  # only `mock` (degraded) answers
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD)
        flushed(client, gw)
        assert "x-cache-layer" not in ask(client, PASSWORD).headers and gw.cache.counters["stored"] == 0
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, "tell me a story of many words", max_tokens=2)  # finish_reason: length
        ask(client, PASSWORD, cache=False)  # client opt-out
        flushed(client, gw)
        assert gw.cache.counters["stored"] == 0


def test_multi_turn_conversations_use_the_exact_layer_only():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    convo = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": PASSWORD},
    ]
    with TestClient(app, headers=make_client().headers) as client:
        client.post(URL, json={"model": "fast", "messages": convo})
        flushed(client, gw)
        assert client.post(URL, json={"model": "fast", "messages": convo}).headers["x-cache-layer"] == "exact"
        near = [*convo[:2], {"role": "user", "content": PASSWORD_PARAPHRASE}]
        assert "x-cache-layer" not in client.post(URL, json={"model": "fast", "messages": near}).headers


def test_a_cached_answer_is_served_as_a_stream_too_and_a_stream_fills_the_cache():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    with TestClient(app, headers=make_client().headers) as client:
        first = ask(client, PASSWORD, stream=True)  # miss: written back when it ends
        flushed(client, gw)
        again = ask(client, PASSWORD, stream=True)
        plain = ask(client, PASSWORD)
        assert again.headers["x-cache-layer"] == "exact" and again.text.strip().endswith("[DONE]")
        text = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            for line in again.text.splitlines()
            if line.startswith("data: {")
        )
        assert text == plain.json()["choices"][0]["message"]["content"] == f"Mock reply to: {PASSWORD}"
        assert plain.headers["x-cache-layer"] == "exact" and groq.calls == 1 and first.status_code == 200


def test_exact_only_aliases_ignore_paraphrases():
    groq = Counting()
    app, gw = cached_app(groq=groq)
    gw.providers["mock"] = groq  # `judge` = ollama (unregistered) -> mock (degraded)
    gw.routes.aliases["judge"].chain[-1].degraded = False
    with TestClient(app, headers=make_client().headers) as client:
        ask(client, PASSWORD, alias="judge")
        flushed(client, gw)
        assert ask(client, PASSWORD, alias="judge").headers["x-cache-layer"] == "exact"
        assert "x-cache-layer" not in ask(client, PASSWORD_PARAPHRASE, alias="judge").headers


# --- /v1/embeddings ---------------------------------------------------------------------------------------


def test_embeddings_endpoint_returns_float_and_base64_vectors():
    app = make_app(embedder="hash")
    client = TestClient(app, headers=make_client().headers)
    floats = client.post("/v1/embeddings", json={"model": "hash", "input": ["a b c", "x y z"]}).json()
    assert [d["index"] for d in floats["data"]] == [0, 1] and len(floats["data"][0]["embedding"]) == 256
    b64 = client.post("/v1/embeddings", json={"model": "hash", "input": "a b c", "encoding_format": "base64"})
    raw = base64.b64decode(b64.json()["data"][0]["embedding"])
    assert list(struct.unpack("<256f", raw)) == pytest.approx(floats["data"][0]["embedding"], abs=1e-6)
    assert floats["usage"]["prompt_tokens"] > 0


def test_embeddings_errors():
    client = TestClient(make_app(embedder="hash"), headers=make_client().headers)
    assert client.post("/v1/embeddings", json={"model": "nope", "input": "x"}).status_code == 404
    assert client.post("/v1/embeddings", json={"model": "hash", "input": []}).status_code == 400
    assert client.post("/v1/embeddings", json={"model": "hash", "input": " "}).status_code == 400
    assert (
        TestClient(make_app(embedder="hash"))
        .post("/v1/embeddings", json={"model": "hash", "input": "x"})
        .status_code
        == 401
    )
    off = TestClient(make_app(), headers={"Authorization": f"Bearer {FULL_KEY}"})
    r = off.post("/v1/embeddings", json={"model": "hash", "input": "x"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "embeddings_unavailable"


def test_stats_reports_the_cache_state():
    app, gw = cached_app()
    snap = TestClient(app).get("/stats").json()["cache"]
    assert snap["backend"] == "memory" and snap["embedder"] == "hash" and snap["semantic"] == "serving"
    assert TestClient(make_app()).get("/stats").json()["cache"] is None
