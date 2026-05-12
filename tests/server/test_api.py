from __future__ import annotations

from fastapi.testclient import TestClient

from src.server.app import create_app
from src.server.config import Settings


class FakeEmbeddingService:
    device = "cuda"

    def __init__(self, ready: bool = True):
        self._ready = ready

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def load(self) -> None:
        self._ready = True

    async def embed(self, resolved, batch_size=None):
        return [[float(item.index), 1.0, float(batch_size or 0)] for item in resolved]


def make_client(api_key: str | None = None, ready: bool = True) -> TestClient:
    settings = Settings(api_key=api_key, load_model_on_startup=False, max_batch_size=4)
    app = create_app(settings=settings, service=FakeEmbeddingService(ready=ready))
    return TestClient(app)


def test_health_does_not_require_ready_model() -> None:
    client = make_client(ready=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_when_model_not_loaded() -> None:
    client = make_client(ready=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "model is not ready"


def test_embeddings_preserve_request_order() -> None:
    client = make_client()
    response = client.post(
        "/v1/embeddings",
        json={
            "inputs": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
            "batch_size": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dimensions"] == 3
    assert payload["data"][0]["index"] == 0
    assert payload["data"][0]["embedding"] == [0.0, 1.0, 2.0]
    assert payload["data"][1]["index"] == 1
    assert payload["data"][1]["embedding"] == [1.0, 1.0, 2.0]


def test_api_key_auth_accepts_bearer_token() -> None:
    client = make_client(api_key="secret")
    response = client.post("/v1/embeddings", json={"inputs": [{"type": "text", "text": "hello"}]})
    assert response.status_code == 401

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer secret"},
        json={"inputs": [{"type": "text", "text": "hello"}]},
    )
    assert response.status_code == 200


def test_batch_size_limit() -> None:
    client = make_client()
    response = client.post(
        "/v1/embeddings",
        json={"inputs": [{"type": "text", "text": "hello"}], "batch_size": 99},
    )
    assert response.status_code == 400
    assert "batch_size" in response.json()["detail"]


def test_upload_endpoint_accepts_payload_without_files_for_text() -> None:
    client = make_client()
    response = client.post(
        "/v1/embeddings/uploads",
        data={"payload": '{"inputs":[{"type":"text","text":"hello"}]}'},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.0, 1.0, 0.0]

