from helpers import make_client


def client():
    return make_client()


def test_healthz():
    r = client().get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_lists_aliases():
    ids = {m["id"] for m in client().get("/v1/models").json()["data"]}
    assert {"fast", "smart", "judge"} <= ids


def test_the_guide_page_is_served_by_the_gateway_with_a_same_origin_csp():
    from helpers import make_client

    r = make_client(key=None).get("/guide")
    assert r.status_code == 200 and "LLM Gateway" in r.text
    assert "connect-src 'self'" in r.headers["content-security-policy"]
