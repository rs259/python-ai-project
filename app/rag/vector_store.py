import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "app/rag/faiss.index"
TEXT_PATH = "app/rag/text_chunks.npy"
METADATA_PATH = "app/rag/metadata.npy"
BUNDLE_PATH = "app/rag/vector_store.npz"
model = SentenceTransformer(MODEL_NAME)

def save_vector_bundle(index, chunks, metadata, bundle_path=None):
    """Save FAISS index, chunks and metadata as one atomic bundle."""
    import tempfile

    bundle_path = bundle_path or BUNDLE_PATH

    if index.ntotal != len(chunks) or index.ntotal != len(metadata):
        raise ValueError("FAISS index, chunks and metadata counts do not match.")

    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(bundle_path),
            suffix=".npz",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

            np.savez(
                temp_file,
                index_bytes=faiss.serialize_index(index),
                chunks=np.asarray(chunks, dtype=object),
                metadata=np.asarray(metadata, dtype=object),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Verify the complete bundle before publishing it.
        with np.load(temp_path, allow_pickle=True) as data:
            check_index = faiss.deserialize_index(data["index_bytes"])
            check_chunks = data["chunks"]
            check_metadata = data["metadata"]

            if not (
                check_index.ntotal
                == len(check_chunks)
                == len(check_metadata)
            ):
                raise ValueError("Temporary bundle verification failed.")

        os.replace(temp_path, bundle_path)
        temp_path = None

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
def create_chunks(text: str, chunk_size: int = 80):
    """Clean text and create chunks while preserving CSV records."""
    import csv
    import io
    import re

    text = text.replace("\x7f", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text:
        return []

    # Handle country/currency CSV tables.
    lines = text.splitlines()
    if (
        lines
        and "country" in lines[0].lower()
        and "currency" in lines[0].lower()
        and "," in lines[0]
    ):
        try:
            rows = csv.DictReader(io.StringIO(text))
            csv_chunks = []

            for row in rows:
                record = ", ".join(
                    f"{key.strip()}: {value.strip()}"
                    for key, value in row.items()
                    if key and value and value.strip()
                )
                if record:
                    csv_chunks.append(record)

            if csv_chunks:
                return csv_chunks
        except (csv.Error, AttributeError):
            pass

    # Separate known headings from surrounding text.
    headings = (
        "Purpose",
        "Python",
        "FastAPI",
        "RAG Pipeline",
        "Ollama and Llama 3.2",
        "Test Facts",
        "Suggested test questions",
    )

    for heading in headings:
        text = re.sub(
            rf"(?<!\w){re.escape(heading)}(?!\w)",
            "\n\n" + heading + "\n",
            text,
        )

    sections = [
        part.strip()
        for part in re.split(r"\n+", text)
        if part.strip()
    ]

    chunks = []
    current = []

    for section in sections:
        words = section.split()

        for i in range(0, len(words), chunk_size):
            piece_words = words[i:i + chunk_size]

            if current and len(current) + len(piece_words) > chunk_size:
                chunks.append(" ".join(current))
                current = []

            current.extend(piece_words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def create_vector_store(
    text: str,
    filename: str = "unknown",
    document_id: int | None = None,
    user_id: int | None = None,
):
    """Append document chunks and atomically save the vector bundle."""

    chunks = create_chunks(text)
    if not chunks:
        raise ValueError("No text found to create vector store.")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    os.makedirs("app/rag", exist_ok=True)

    # Prefer the bundle when it already exists.
    if os.path.exists(BUNDLE_PATH):
        with np.load(BUNDLE_PATH, allow_pickle=True) as data:
            index = faiss.deserialize_index(data["index_bytes"])
            old_chunks = data["chunks"].tolist()
            old_metadata = data["metadata"].tolist()
    elif (
        os.path.exists(INDEX_PATH)
        and os.path.exists(TEXT_PATH)
        and os.path.exists(METADATA_PATH)
    ):
        # One-time migration path from the legacy files.
        index = faiss.read_index(INDEX_PATH)
        old_chunks = np.load(TEXT_PATH, allow_pickle=True).tolist()
        old_metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
    else:
        index = faiss.IndexFlatL2(dimension)
        old_chunks = []
        old_metadata = []

    if index.d != dimension:
        raise ValueError(
            f"FAISS dimension mismatch. Existing: {index.d}, New: {dimension}"
        )

    if not (index.ntotal == len(old_chunks) == len(old_metadata)):
        raise ValueError(
            "Existing FAISS index, chunks and metadata counts do not match."
        )

    start_chunk_index = len(old_metadata)

    new_metadata = [
        {
            "filename": filename,
            "document_id": document_id,
            "user_id": user_id,
            "chunk_index": start_chunk_index + i,
        }
        for i in range(len(chunks))
    ]

    combined_chunks = old_chunks + chunks
    combined_metadata = old_metadata + new_metadata

    # Build a fresh index so the current in-memory index is not mutated.
    combined_embeddings = []

    if index.ntotal:
        # Reconstruct the stored vectors from the existing flat L2 index.
        if not isinstance(index, faiss.IndexFlat):
            raise ValueError("Only flat FAISS indexes are supported here.")

        old_vectors = index.reconstruct_n(0, index.ntotal)
        combined_embeddings.append(old_vectors)

    combined_embeddings.append(embeddings)
    vectors = np.vstack(combined_embeddings).astype("float32")

    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(vectors)

    save_vector_bundle(new_index, combined_chunks, combined_metadata)

    return {
        "chunks_added": len(chunks),
        "total_chunks": len(combined_chunks),
        "dimension": dimension,
        "filename": filename,
        "document_id": document_id,
        "user_id": user_id,
    }
def search_vector_store(
    query: str,
    top_k: int = 3,
    user_id: int | None = None,
    bundle_path=None,
):
    """Search FAISS and return unique, user-specific results."""

    if user_id is None:
        raise ValueError("user_id is required for document search.")

    bundle_path = bundle_path or BUNDLE_PATH

    if os.path.exists(bundle_path):
        with np.load(bundle_path, allow_pickle=True) as data:
            index = faiss.deserialize_index(data["index_bytes"])
            chunks = data["chunks"].copy()
            metadata = data["metadata"].copy()
    else:
        for required_path, message in (
            (INDEX_PATH, "FAISS index not found. Please upload a document first."),
            (TEXT_PATH, "Text chunks not found. Please upload a document first."),
            (METADATA_PATH, "Metadata not found. Please rebuild the vector store."),
        ):
            if not os.path.exists(required_path):
                raise FileNotFoundError(message)

        index = faiss.read_index(INDEX_PATH)
        chunks = np.load(TEXT_PATH, allow_pickle=True)
        metadata = np.load(METADATA_PATH, allow_pickle=True)

    if index.ntotal != len(chunks) or index.ntotal != len(metadata):
        raise ValueError("FAISS index, chunks and metadata counts do not match.")

    if index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    distances, indices = index.search(query_embedding, index.ntotal)

    results = []
    seen_texts = set()

    for distance, index_id in zip(distances[0], indices[0]):
        if index_id < 0:
            continue

        meta = metadata[index_id]

        if meta.get("user_id") != user_id:
            continue

        distance = float(distance)
        if distance > 2.0:
            continue

        chunk_text = str(chunks[index_id])
        normalized_text = " ".join(chunk_text.split()).casefold()

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)
        results.append({
            "text": chunk_text,
            "distance": distance,
            "filename": meta["filename"],
            "document_id": meta.get("document_id"),
            "user_id": meta.get("user_id"),
            "chunk_index": int(meta["chunk_index"]),
        })

    results.sort(key=lambda item: item["distance"])
    return results[:max(1, top_k)]

def rebuild_vector_store(documents, bundle_path=None):
    """Rebuild the complete Knowledge Base and atomically save its bundle."""
    from pathlib import Path
    from app.rag.document_loader import extract_text_from_file

    all_chunks = []
    all_metadata = []
    all_embeddings = []
    indexed_paths = set()
    indexed_documents = 0

    for document in documents:
        path = Path(document.file_path).resolve()

        if path in indexed_paths:
            continue
        indexed_paths.add(path)

        if not path.is_file():
            continue

        text = extract_text_from_file(str(path))
        chunks = create_chunks(text)

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
        ).astype("float32")

        all_embeddings.append(embeddings)
        indexed_documents += 1

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "filename": document.filename,
                "document_id": document.id,
                "user_id": document.user_id,
                "chunk_index": len(all_metadata),
            })

    os.makedirs("app/rag", exist_ok=True)

    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings).astype("float32")
        index = faiss.IndexFlatL2(combined_embeddings.shape[1])
        index.add(combined_embeddings)
    else:
        # Empty index: keep its dimension consistent with the embedding model.
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)

    save_vector_bundle(index, all_chunks, all_metadata, bundle_path=bundle_path)

    return {
        "total_chunks": len(all_chunks),
        "total_documents": indexed_documents,
    }
