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
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_store(text: str, filename: str = "unknown"):
    """
    Add document chunks to existing FAISS index.
    New documents are appended instead of overwriting old documents.
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

    # --------------------------------------------------
    # Create new index OR load existing index
    # --------------------------------------------------

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        if index.d != dimension:
            raise ValueError(
                "Embedding dimension does not match existing FAISS index."
            )

    else:
        index = faiss.IndexFlatL2(dimension)

    # --------------------------------------------------
    # Add new embeddings
    # --------------------------------------------------

    index.add(embeddings)

    os.makedirs("app/rag", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    # --------------------------------------------------
    # Existing text chunks
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
    # Metadata
    # --------------------------------------------------

    if os.path.exists(METADATA_PATH):
        old_metadata = np.load(
            METADATA_PATH,
            allow_pickle=True
        ).tolist()
    else:
        old_metadata = []

    for _ in chunks:
        old_metadata.append({
            "filename": filename
        })

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

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Please upload a document first."
        )

    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(
            "Text chunks not found. Please upload a document first."
        )

    index = faiss.read_index(INDEX_PATH)

    chunks = np.load(
        TEXT_PATH,
        allow_pickle=True
    )

    metadata = []

    if os.path.exists(METADATA_PATH):
        metadata = np.load(
            METADATA_PATH,
            allow_pickle=True
        )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        item = {
            "text": str(chunks[index_id]),
            "distance": float(distance)
        }

        if len(metadata) > index_id:
            item["filename"] = metadata[index_id]["filename"]
        else:
            item["filename"] = "unknown"

        results.append(item)

    return results
