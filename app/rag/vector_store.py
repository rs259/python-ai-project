import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "app/rag/faiss.index"
TEXT_PATH = "app/rag/text_chunks.npy"
METADATA_PATH = "app/rag/metadata.npy"

model = SentenceTransformer(MODEL_NAME)



def create_chunks(text: str, chunk_size: int = 80):
    """Split country records into individual chunks; chunk other text by words."""
    import csv
    import io
    import re

    text = text.strip()
    if not text:
        return []

    lines = text.splitlines()
    chunks = []

    # Detect CSV-style country tables.
    if (
        lines
        and "Country" in lines[0]
        and "Currency" in lines[0]
        and "," in lines[0]
    ):
        try:
            rows = list(csv.DictReader(io.StringIO(text)))

            if rows:
                for row in rows:
                    record = ", ".join(
                        f"{key.strip()}: {value.strip()}"
                        for key, value in row.items()
                        if key and value and value.strip()
                    )
                    if record:
                        chunks.append(record)

                return chunks
        except (csv.Error, AttributeError):
            pass

    # Split text containing Country: headings into separate sections.
    sections = re.split(r"(?=Country:\s*)", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        words = section.split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

    return chunks

def create_vector_store(
    text: str,
    filename: str = "unknown",
    document_id: int | None = None,
    user_id: int | None = None,
):
    """
    Add a document to the existing FAISS index.

    New documents are appended.
    Existing documents are not overwritten.
    """

    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No text found to create vector store.")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    os.makedirs("app/rag", exist_ok=True)

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        if index.d != dimension:
            raise ValueError(
                f"FAISS dimension mismatch. Existing: {index.d}, New: {dimension}"
            )
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

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
        new_metadata.append({
            "filename": filename,
            "document_id": document_id,
            "user_id": user_id,
            "chunk_index": start_chunk_index + i,
        })

    old_metadata.extend(new_metadata)

    np.save(
        METADATA_PATH,
        np.array(old_metadata, dtype=object)
    )

    return {
        "chunks_added": len(chunks),
        "total_chunks": len(old_chunks),
        "dimension": dimension,
        "filename": filename,
        "document_id": document_id,
        "user_id": user_id,
    }


def search_vector_store(
    query: str,
    top_k: int = 3,
    user_id: int | None = None,
):
    """Search FAISS and return the best matching chunks for a user."""

    if user_id is None:
        raise ValueError("user_id is required for document search.")

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
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, index.ntotal)

    results = []
    for distance, index_id in zip(distances[0], indices[0]):
        if index_id < 0:
            continue

        meta = metadata[index_id]

        if meta.get("user_id") != user_id:
            continue

        results.append({
            "text": str(chunks[index_id]),
            "distance": float(distance),
            "filename": meta["filename"],
            "document_id": meta.get("document_id"),
            "user_id": meta.get("user_id"),
            "chunk_index": int(meta["chunk_index"]),
        })


    results.sort(key=lambda item: item["distance"])

    # Reject weak matches
    MAX_DISTANCE = 1.00
    results = [
        item for item in results
        if item["distance"] <= MAX_DISTANCE
    ]

    return results[:max(1, top_k)]

def rebuild_vector_store(documents):
    """Rebuild FAISS, indexing each existing physical file only once."""
    import os
    from pathlib import Path
    from app.rag.document_loader import extract_text_from_file

    all_chunks = []
    all_metadata = []
    all_embeddings = []
    seen_paths = set()
    indexed_documents = 0

    for document in documents:
        path = Path(document.file_path).resolve()

        if path in seen_paths:
            continue
        seen_paths.add(path)

        if not path.is_file():
            continue

        text = extract_text_from_file(str(path))
        chunks = create_chunks(text)

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            convert_to_numpy=True
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
        combined_embeddings = np.vstack(all_embeddings)
        index = faiss.IndexFlatL2(combined_embeddings.shape[1])
        index.add(combined_embeddings)
        faiss.write_index(index, INDEX_PATH)
    elif os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    np.save(TEXT_PATH, np.array(all_chunks, dtype=object))
    np.save(METADATA_PATH, np.array(all_metadata, dtype=object))

    return {
        "total_chunks": len(all_chunks),
        "total_documents": indexed_documents,
    }
