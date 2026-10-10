from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    response = client.post(
        "/predict",
        json={"features": [0.0] * 30}
    )
    assert response.status_code == 200
