from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from jose import jwt

from app.routes.auth import (
    verify_token,
    SECRET_KEY,
    ALGORITHM,
)

app = FastAPI()


@app.get("/protected")
def protected(user_id: int = Depends(verify_token)):
    return {"user_id": user_id}


client = TestClient(app)


def test_missing_token_is_rejected():
    response = client.get("/protected")
    assert response.status_code == 401


def test_invalid_token_is_rejected():
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_token_without_subject_is_rejected():
    token = jwt.encode({}, SECRET_KEY, algorithm=ALGORITHM)
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_expired_token_is_rejected():
    token = jwt.encode(
        {
            "sub": "123",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_valid_token_is_accepted():
    token = jwt.encode(
        {
            "sub": "123",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == {"user_id": 123}
