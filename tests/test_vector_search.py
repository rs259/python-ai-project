import faiss
import numpy as np

from app.rag.vector_store import (
    save_vector_bundle,
    search_vector_store,
)


def test_faiss_search_prioritizes_python_document(tmp_path, monkeypatch):
    import app.rag.vector_store as vector_store

    monkeypatch.setattr(
        vector_store.model,
        "encode",
        lambda texts, **kwargs: np.array(
            [[1.0, 0.0]], dtype="float32"
        ),
    )

    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1.0, 0.0]], dtype="float32"))

    chunks = [
        "Python AI RAG project uses FAISS for vector search."
    ]
    metadata = [
        {
            "filename": "Python_AI_RAG_Test_Document.pdf",
            "document_id": 1,
            "user_id": 2,
            "chunk_index": 0,
        }
    ]

    bundle_path = tmp_path / "test_vector_store.npz"
    save_vector_bundle(
        index,
        chunks,
        metadata,
        bundle_path=str(bundle_path),
    )

    results = search_vector_store(
        "What is FAISS?",
        top_k=3,
        user_id=2,
        bundle_path=str(bundle_path),
    )

    assert results
    assert results[0]["filename"] == "Python_AI_RAG_Test_Document.pdf"
    assert "faiss" in results[0]["text"].lower()


def test_user_cannot_search_another_users_document(tmp_path, monkeypatch):
    import app.rag.vector_store as vector_store

    monkeypatch.setattr(
        vector_store.model,
        "encode",
        lambda texts, **kwargs: np.array(
            [[1.0, 0.0]], dtype="float32"
        ),
    )

    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1.0, 0.0]], dtype="float32"))

    secret_phrase = "CONFIDENTIAL ALPHA ORCHID 73921"
    chunks = [secret_phrase]
    metadata = [
        {
            "filename": "private_user_a.txt",
            "document_id": 101,
            "user_id": 2,
            "chunk_index": 0,
        }
    ]

    bundle_path = tmp_path / "user_isolation_test.npz"
    save_vector_bundle(
        index,
        chunks,
        metadata,
        bundle_path=str(bundle_path),
    )

    # User A should find their own document.
    user_a_results = search_vector_store(
        secret_phrase,
        top_k=3,
        user_id=2,
        bundle_path=str(bundle_path),
    )

    # User B must not receive User A's document.
    user_b_results = search_vector_store(
        secret_phrase,
        top_k=3,
        user_id=3,
        bundle_path=str(bundle_path),
    )

    assert any(
        secret_phrase in result["text"]
        for result in user_a_results
    )
    assert all(
        secret_phrase not in result["text"]
        for result in user_b_results
    )
    assert user_b_results == []
