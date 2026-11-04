import pytest
from fastapi.testclient import TestClient
from helpers import DEMO_KEY, FULL_KEY, NO_ALIASES_KEY, make_app, make_client

from llm_gateway.providers.mock import MockProvider
from llm_gateway.security.clients import ClientConfig, ClientRegistry

MSG = [{"role": "user", "content": "hola"}]
BODY = {"model": "fast", "messages": MSG}


class CountingProvider(MockProvider):
    calls = 0

    async def complete(self, request, model):
        type(self).calls += 1
        return await super().complete(request, model)


def counting_client(key: str | None) -> TestClient:
    CountingProvider.calls = 0
    app = make_app()
    app.state.gateway.providers["mock"] = CountingProvider()
    client = TestClient(app)
    if key:
        client.headers["Authorization"] = f"Bearer {key}"
    return client


@pytest.mark.parametrize(
    "headers,code",
    [
        ({}, "missing_api_key"),
        ({"Authorization": "Bearer"}, "missing_api_key"),
        ({"Authorization": "Basic abc"}, "missing_api_key"),
        ({"Authorization": "Bearer gw_full_wrong"}, "invalid_api_key"),
    ],
)
def test_req_02_bad_or_missing_key_is_401_and_calls_no_provider(headers, code):
    client = counting_client(None)
    r = client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert r.status_code == 401 and r.json()["error"]["code"] == code
    assert r.headers["www-authenticate"] == "Bearer"
    assert CountingProvider.calls == 0


def test_valid_key_reaches_the_provider():
    client = counting_client(FULL_KEY)
    assert client.post("/v1/chat/completions", json=BODY).status_code == 200
    assert CountingProvider.calls == 1


def test_req_18_alias_not_allowed_is_403_and_calls_no_provider():
    client = counting_client(DEMO_KEY)
    r = client.post("/v1/chat/completions", json={**BODY, "model": "smart"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "alias_not_allowed"
    assert CountingProvider.calls == 0
    assert client.post("/v1/chat/completions", json=BODY).status_code == 200


def test_empty_allowed_aliases_is_fail_closed():
    r = make_client(key=NO_ALIASES_KEY).post("/v1/chat/completions", json=BODY)
    assert r.status_code == 403


def test_models_lists_only_the_aliases_the_client_may_use():
    assert [m["id"] for m in make_client(key=DEMO_KEY).get("/v1/models").json()["data"]] == ["fast"]
    assert "smart_paid" in {m["id"] for m in make_client().get("/v1/models").json()["data"]}
    assert make_client(key=None).get("/v1/models").status_code == 401


def test_unknown_alias_message_lists_only_the_clients_aliases():
    r = make_client(key=DEMO_KEY).post("/v1/chat/completions", json={**BODY, "model": "nope"})
    assert r.status_code == 404 and "fast" in r.json()["error"]["message"]
    assert "smart_paid" not in r.json()["error"]["message"]


def test_healthz_stays_public():
    assert make_client(key=None).get("/healthz").status_code == 200


def test_missing_clients_file_rejects_everything(tmp_path):
    from helpers import PROVIDERS_FILE

    from llm_gateway.main import create_app
    from llm_gateway.settings import Settings

    app = create_app(
        Settings(
            _env_file=None, env="test", clients_file=tmp_path / "absent.yaml", providers_file=PROVIDERS_FILE
        )
    )
    headers = {"Authorization": f"Bearer {FULL_KEY}"}
    r = TestClient(app).post("/v1/chat/completions", json=BODY, headers=headers)
    assert r.status_code == 401


def test_client_config_rejects_a_placeholder_hash():
    with pytest.raises(ValueError, match="64 hex"):
        ClientConfig(key_sha256="<sha256>")


def test_two_clients_cannot_share_a_key():
    same = ClientConfig(key_sha256="a" * 64)
    with pytest.raises(ValueError, match="share"):
        ClientRegistry({"a": same, "b": same})


def test_unknown_alias_in_clients_file_stops_startup():
    from pathlib import Path

    from llm_gateway.routing.config import load_routes

    registry = ClientRegistry({"x": ClientConfig(key_sha256="b" * 64, allowed_aliases=["fats"])})
    with pytest.raises(ValueError, match="fats"):
        registry.check_routes(load_routes(Path("config/routes.yaml")))
