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
