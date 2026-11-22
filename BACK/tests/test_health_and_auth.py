# Tests for GET /health and the proxy shared-secret check on the chat
# endpoint (tmpbot_backend.main, tmpbot_backend.auth).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Health probe and shared-secret tests."""

from __future__ import annotations

import dataclasses

from chat_helpers import build_runtime
from fastapi.testclient import TestClient

from tmpbot_backend import __version__
from tmpbot_backend.main import create_app


def test_health_ok_without_secret(client):
    """Health needs no secret: Render's probe sends no headers."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok", "version": __version__, "env": "local",
    }


def test_health_sets_request_id(client):
    """Every response carries an X-Request-Id header."""
    resp = client.get("/health")
    assert len(resp.headers["x-request-id"]) >= 8


def test_request_id_from_proxy_is_echoed(client):
    """A well-formed incoming request id is kept for log correlation."""
    resp = client.get("/health", headers={"X-Request-Id": "abc-12345678"})
    assert resp.headers["x-request-id"] == "abc-12345678"


def test_malformed_request_id_is_replaced(client):
    """A request id with odd characters is not echoed into logs."""
    resp = client.get("/health", headers={"X-Request-Id": "bad id\twith"})
    assert resp.headers["x-request-id"] != "bad id\twith"


def test_chat_without_secret_is_401(client, make_body):
    """No header: refused."""
    resp = client.post("/api/v1/chat", json=make_body())
    assert resp.status_code == 401


def test_chat_with_wrong_secret_is_401(client, make_body):
    """Wrong header value: refused."""
    resp = client.post(
        "/api/v1/chat",
        json=make_body(),
        headers={"X-TMPBOT-Proxy-Secret": "not-the-secret"},
    )
    assert resp.status_code == 401


def test_chat_with_empty_secret_header_is_401(client, make_body):
    """An empty header never matches."""
    resp = client.post(
        "/api/v1/chat",
        json=make_body(),
        headers={"X-TMPBOT-Proxy-Secret": ""},
    )
    assert resp.status_code == 401


def test_chat_fails_closed_when_backend_secret_unset(settings, make_body):
    """A backend deployed without TMPBOT_SHARED_SECRET refuses everyone,
    even a caller that sends an empty header."""
    app = create_app(dataclasses.replace(settings, shared_secret=""))
    client = TestClient(app)
    for headers in ({}, {"X-TMPBOT-Proxy-Secret": ""},
                    {"X-TMPBOT-Proxy-Secret": "anything"}):
        resp = client.post("/api/v1/chat", json=make_body(), headers=headers)
        assert resp.status_code == 401, headers


def test_auth_checked_before_body_validation(client):
    """A caller without the secret learns nothing about the body schema."""
    resp = client.post("/api/v1/chat", json={"nonsense": True})
    assert resp.status_code == 401


def test_chat_with_right_secret_is_200(client, make_body, auth_headers):
    """Correct header: the stream starts."""
    resp = client.post("/api/v1/chat", json=make_body(), headers=auth_headers)
    assert resp.status_code == 200


def test_docs_hidden_in_production(settings):
    """The API schema is not published in production."""
    prod = dataclasses.replace(settings, env="production")
    app = create_app(prod, build_runtime(prod))
    client = TestClient(app)
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/health").json()["env"] == "production"
