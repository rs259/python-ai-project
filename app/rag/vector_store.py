import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "app/rag/faiss.index"
TEXT_PATH = "app/rag/text_chunks.npy"


model = SentenceTransformer(MODEL_NAME)


def create_chunks(text: str, chunk_size: int = 500):
    """
    Split text into small chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_store(text: str):
    """
    Create FAISS vector index from text.
    """

    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No text found to create vector store.")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    os.makedirs("app/rag", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    np.save(
        TEXT_PATH,
        np.array(chunks, dtype=object)
    )

    return {
        "chunks": len(chunks),
        "dimension": dimension
    }


def search_vector_store(query: str, top_k: int = 3):
    """
    Search similar chunks from FAISS.
    """

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found.")

    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError("Text chunks not found.")

    index = faiss.read_index(INDEX_PATH)

    chunks = np.load(
        TEXT_PATH,
        allow_pickle=True
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):
        if index_id == -1:
            continue

        results.append({
            "text": str(chunks[index_id]),
            "distance": float(distance)
        })

    return results
