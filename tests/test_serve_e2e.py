"""E2E tests for ticket_router_serve FastAPI application."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


API_KEY = "test-api-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client(monkeypatch):
    """Create test client with mocked startup (skip model loading)."""
    import ticket_router.serve.deps

    # Patch API_KEYS so verify_api_key accepts our test key
    monkeypatch.setattr(ticket_router.serve.deps, "API_KEYS", {API_KEY})

    # Patch model pool initialization to skip loading real models
    with patch("ticket_router_serve.main.get_pool") as mock_get_pool:
        mock_instance = MagicMock()
        mock_instance.initialize = MagicMock()
        mock_get_pool.return_value = mock_instance
        from ticket_router.serve.main import app

        with TestClient(app, raise_server_exceptions=True) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_with_valid_api_key(client):
    """Health endpoint returns 200 with valid API key."""
    response = client.get("/health", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_without_api_key_rejects(client):
    """Health endpoint returns 401 without API key."""
    response = client.get("/health")
    assert response.status_code == 401


def test_health_with_invalid_api_key_rejects(client):
    """Health endpoint returns 401 with invalid API key."""
    response = client.get("/health", headers={"X-API-Key": "bad-key"})
    assert response.status_code == 401


def test_predict_without_api_key_rejects(client):
    """POST /predict returns 401 without API key."""
    response = client.post(
        "/predict",
        json={"body": "test body", "model": "lr"},
    )
    assert response.status_code == 401


def test_predict_unknown_model_returns_400(client):
    """POST /predict with unknown model returns 400."""
    response = client.post(
        "/predict",
        json={"body": "test body", "model": "unknown-model"},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]["error"]


def test_predict_valid_request_returns_req_id(client):
    """POST /predict with valid request returns req_id and PENDING status."""
    with patch("ticket_router_serve.main.submit_task") as mock_submit:
        mock_submit.return_value = "test-req-id-123"
        response = client.post(
            "/predict",
            json={"body": "test body", "model": "lr"},
            headers=HEADERS,
        )
    data = response.json()
    assert data["req_id"] == "test-req-id-123"
    assert data["status"] == "PENDING"
    assert data["cached"] is False


def test_predict_cache_hit_returns_cached(client):
    """POST /predict with same body+model returns cached req_id."""
    fingerprint = "abc123fingerprint"
    cached_req_id = str(uuid.uuid4())

    with patch("ticket_router_serve.main.compute_fingerprint", return_value=fingerprint):
        with patch("ticket_router_serve.main.find_by_fingerprint", return_value=cached_req_id):
            with patch(
                "ticket_router_serve.main.get_cache_entry",
                return_value={"status": "COMPLETED"},
            ):
                response = client.post(
                    "/predict",
                    json={"body": "same body", "model": "lr"},
                    headers=HEADERS,
                )
    data = response.json()
    assert data["req_id"] == cached_req_id
    assert data["status"] == "COMPLETED"
    assert data["cached"] is True


def test_result_without_api_key_rejects(client):
    """GET /result/{req_id} returns 401 without API key."""
    response = client.get("/result/some-req-id")
    assert response.status_code == 401


def test_result_not_found_returns_404(client):
    """GET /result/{req_id} returns 404 for unknown req_id."""
    with patch("ticket_router_serve.main.get_cache_entry", return_value=None):
        response = client.get("/result/unknown-req-id", headers=HEADERS)
    assert response.status_code == 404


def test_result_pending_returns_pending_status(client):
    """GET /result/{req_id} returns PENDING status for pending task."""
    with patch(
        "ticket_router_serve.main.get_cache_entry",
        return_value={"req_id": "test-id", "status": "PENDING"},
    ):
        response = client.get("/result/test-id", headers=HEADERS)
    data = response.json()
    assert data["req_id"] == "test-id"
    assert data["status"] == "PENDING"
    assert data["result"] is None


def test_result_completed_returns_result(client):
    """GET /result/{req_id} returns result for completed task."""
    mock_result = {
        "queue": "Technical Support",
        "priority": "high",
        "answer": "Please reset your password.",
        "confidence": {"queue": 0.92, "priority": 0.78},
    }
    with patch(
        "ticket_router_serve.main.get_cache_entry",
        return_value={"req_id": "test-id", "status": "COMPLETED", "result": mock_result},
    ):
        response = client.get("/result/test-id", headers=HEADERS)
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["result"]["queue"] == "Technical Support"
    assert data["result"]["priority"] == "high"


def test_attribution_without_api_key_rejects(client):
    """GET /attribution/{req_id} returns 401 without API key."""
    response = client.get("/attribution/some-req-id")
    assert response.status_code == 401


def test_attribution_not_found_returns_404(client):
    """GET /attribution/{req_id} returns 404 for unknown req_id."""
    with patch("ticket_router_serve.main.get_cache_entry", return_value=None):
        response = client.get("/attribution/unknown-id", headers=HEADERS)
    assert response.status_code == 404


def test_attribution_not_completed_returns_none(client):
    """GET /attribution/{req_id} returns None attribution for PENDING task."""
    with patch(
        "ticket_router_serve.main.get_cache_entry",
        return_value={"req_id": "test-id", "status": "PENDING", "model": "rembert"},
    ):
        response = client.get("/attribution/test-id", headers=HEADERS)
    data = response.json()
    assert data["status"] == "PENDING"
    assert data["attribution"] is None


def test_attribution_unsupported_model_returns_none(client):
    """GET /attribution/{req_id} returns None for unsupported model."""
    with patch(
        "ticket_router_serve.main.get_cache_entry",
        return_value={"req_id": "test-id", "status": "COMPLETED", "model": "lr"},
    ):
        response = client.get("/attribution/test-id", headers=HEADERS)
    data = response.json()
    assert data["attribution"] is None


def test_attribution_cached_returns_attribution(client):
    """GET /attribution/{req_id} returns cached attribution."""
    mock_attribution = {
        "queue": {
            "predicted_label": "Technical Support",
            "confidence": 0.92,
            "top_positive": [{"token": "password", "score": 0.42}],
            "top_negative": [],
        },
        "priority": None,
    }
    with patch(
        "ticket_router_serve.main.get_cache_entry",
        return_value={
            "req_id": "test-id",
            "status": "COMPLETED",
            "model": "rembert",
            "attribution": mock_attribution,
        },
    ):
        response = client.get("/attribution/test-id", headers=HEADERS)
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["attribution"] is not None
    assert data["attribution"]["queue"]["predicted_label"] == "Technical Support"