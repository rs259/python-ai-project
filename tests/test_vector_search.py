def test_faiss_search_prioritizes_python_document():
    from app.rag.vector_store import search_vector_store

    results = search_vector_store(
        "What is FAISS?",
        top_k=3,
        user_id=2,
    )

    assert results, "Expected search results for FAISS query"

    first_result = results[0]

    assert first_result["filename"] == "Python_AI_RAG_Test_Document.pdf"
    assert "faiss" in first_result["text"].lower()
