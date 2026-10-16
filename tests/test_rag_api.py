from fastapi.testclient import TestClient


def test_health_endpoint():
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200

def test_faiss_question_returns_document_fallback(monkeypatch):
    import app.routes.rag as rag
    from fastapi.testclient import TestClient
    from app.main import app

    # Simulate an authenticated user.
    app.dependency_overrides[rag.verify_token] = lambda: 2

    # Return a relevant chunk without a FAISS definition.
    monkeypatch.setattr(
        rag,
        "search_vector_store",
        lambda *args, **kwargs: [
            {
                "text": "Store vectors in a vector database or FAISS index.",
                "metadata": {"user_id": 2},
                "distance": 0.5,
            }
        ],
    )

    # Avoid calling the actual Ollama model.
    monkeypatch.setattr(
        rag,
        "chat",
        lambda **kwargs: {
            "message": {
                "content": "FAISS stands for Facebook AI Similarity Search."
            }
        },
    )

    monkeypatch.setattr(rag, "save_chat_history", lambda **kwargs: None)

    try:
        client = TestClient(app)
        response = client.get(
            "/rag/ask",
            params={"query": "What is FAISS?"},
        )

        assert response.status_code == 200
        assert response.json()["answer"] == (
            "I could not find the answer in the uploaded document."
        )
    finally:
        app.dependency_overrides.pop(rag.verify_token, None)


def test_rag_answer_uses_document_context(monkeypatch):
    import app.routes.rag as rag
    from fastapi.testclient import TestClient
    from app.main import app

    app.dependency_overrides[rag.verify_token] = lambda: 2

    monkeypatch.setattr(
        rag,
        "search_vector_store",
        lambda *args, **kwargs: [
            {
                "text": "The purpose of RAG is to retrieve relevant document chunks and send them to a language model.",
                "distance": 0.5,
                "user_id": 2,
            }
        ],
    )

    monkeypatch.setattr(
        rag,
        "chat",
        lambda **kwargs: {
            "message": {
                "content": "RAG retrieves relevant document chunks and sends them to a language model."
            }
        },
    )

    monkeypatch.setattr(rag, "save_chat_history", lambda **kwargs: None)

    try:
        client = TestClient(app)
        response = client.get(
            "/rag/ask",
            params={"query": "What is the purpose of RAG?"},
        )

        assert response.status_code == 200
        assert "relevant document chunks" in response.json()["answer"].lower()
    finally:
        app.dependency_overrides.pop(rag.verify_token, None)


def test_nepal_currency_answer_from_document(monkeypatch):
    import app.routes.rag as rag
    from fastapi.testclient import TestClient
    from app.main import app

    app.dependency_overrides[rag.verify_token] = lambda: 2

    monkeypatch.setattr(
        rag,
        "search_vector_store",
        lambda *args, **kwargs: [
            {
                "text": "Nepal Nepalese Rupee Rs NPR",
                "distance": 0.4,
                "user_id": 2,
            }
        ],
    )

    monkeypatch.setattr(rag, "save_chat_history", lambda **kwargs: None)

    try:
        client = TestClient(app)
        response = client.get(
            "/rag/ask",
            params={"query": "What is the currency of Nepal?"},
        )

        assert response.status_code == 200
        assert "Nepalese Rupee" in response.json()["answer"]
        assert "NPR" in response.json()["answer"]
    finally:
        app.dependency_overrides.pop(rag.verify_token, None)
