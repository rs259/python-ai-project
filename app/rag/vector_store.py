import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "app/rag/faiss.index"
TEXT_PATH = "app/rag/text_chunks.npy"
METADATA_PATH = "app/rag/metadata.npy"

model = SentenceTransformer(MODEL_NAME)


def create_chunks(text: str, chunk_size: int = 300):
    """
    Split document text into small chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_store(text: str, filename: str = "unknown"):
    """
    Add a document to the existing FAISS index.

    Important:
    New documents are APPENDED.
    Existing documents are NOT overwritten.
    """

    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No text found to create vector store.")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    os.makedirs("app/rag", exist_ok=True)

    # --------------------------------------------------
    # Load existing FAISS index or create new one
    # --------------------------------------------------

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        # Safety check
        if index.d != dimension:
            raise ValueError(
                f"FAISS dimension mismatch. "
                f"Existing: {index.d}, New: {dimension}"
            )

    else:
        index = faiss.IndexFlatL2(dimension)

    # --------------------------------------------------
    # Add new embeddings
    # --------------------------------------------------

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    # --------------------------------------------------
    # Load existing text chunks
    # --------------------------------------------------

    if os.path.exists(TEXT_PATH):
        old_chunks = np.load(
            TEXT_PATH,
            allow_pickle=True
        ).tolist()
    else:
        old_chunks = []

    old_chunks.extend(chunks)

    np.save(
        TEXT_PATH,
        np.array(old_chunks, dtype=object)
    )

    # --------------------------------------------------
    # Load existing metadata
    # --------------------------------------------------

    if os.path.exists(METADATA_PATH):
        old_metadata = np.load(
            METADATA_PATH,
            allow_pickle=True
        ).tolist()
    else:
        old_metadata = []

    start_chunk_index = len(old_metadata)

    new_metadata = []

    for i, chunk in enumerate(chunks):

        new_metadata.append(
            {
                "filename": filename,
                "chunk_index": start_chunk_index + i
            }
        )

    old_metadata.extend(new_metadata)

    np.save(
        METADATA_PATH,
        np.array(old_metadata, dtype=object)
    )

    return {
        "chunks_added": len(chunks),
        "total_chunks": len(old_chunks),
        "dimension": dimension,
        "filename": filename
    }


def search_vector_store(query: str, top_k: int = 10):
    """
    Search across ALL uploaded documents.
    """

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please upload a document first."
        )

    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(
            "Text chunks not found. Please upload a document first."
        )

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Metadata not found. Please rebuild the vector store."
        )

    index = faiss.read_index(INDEX_PATH)

    chunks = np.load(
        TEXT_PATH,
        allow_pickle=True
    )

    metadata = np.load(
        METADATA_PATH,
        allow_pickle=True
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    # Never request more results than available vectors
    actual_top_k = min(top_k, index.ntotal)

    distances, indices = index.search(
        query_embedding,
        actual_top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        chunk = str(chunks[index_id])

        meta = metadata[index_id]

        results.append(
            {
                "text": chunk,
                "distance": float(distance),
                "filename": meta["filename"],
                "chunk_index": int(meta["chunk_index"])
            }
        )

    return results
