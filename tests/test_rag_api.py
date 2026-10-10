from fastapi.testclient import TestClient


def test_health_endpoint():
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
