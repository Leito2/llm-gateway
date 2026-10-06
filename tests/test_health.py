from fastapi.testclient import TestClient

from llm_gateway.main import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_healthz():
    r = client().get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_lists_aliases():
    ids = {m["id"] for m in client().get("/v1/models").json()["data"]}
    assert {"fast", "smart", "judge"} <= ids
