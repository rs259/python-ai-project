import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "app/rag/faiss.index"
TEXT_PATH = "app/rag/text_chunks.npy"
METADATA_PATH = "app/rag/metadata.npy"
BUNDLE_PATH = "app/rag/vector_store.npz"
_model = None

def get_model():
    """Load the embedding model only when first needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class LazyModel:
    """Load the embedding model on demand and allow tests to mock encode."""
    def __init__(self):
        self._encode_override = None

    def encode(self, *args, **kwargs):
        if self._encode_override is not None:
            return self._encode_override(*args, **kwargs)
        return get_model().encode(*args, **kwargs)

    def __setattr__(self, name, value):
        if name == "encode" and callable(value):
            object.__setattr__(self, "_encode_override", value)
        else:
            object.__setattr__(self, name, value)

    def get_sentence_embedding_dimension(self):
        return get_model().get_sentence_embedding_dimension()


model = LazyModel()

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
    """Clean text and create useful chunks for documents and currency tables."""
    import re

    text = text.replace("\x7f", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    if not text:
        return []

    # Detect the extracted world-currency PDF, where each field may
    # appear on a separate line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)

    if (
        "List of World Currency Names" in joined
        or (
            "Country Currency Name Symbol ISO Code".lower()
            in re.sub(r"\s+", " ", joined).lower()
        )
    ):
        country_names = [
            "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
            "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
            "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
            "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
            "Bosnia & Herzegovina", "Botswana", "Brazil", "Brunei",
            "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon",
            "Canada", "Cape Verde", "Central African Republic", "Chad",
            "Chile", "China", "Colombia", "Comoros", "Congo (DRC)",
            "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
            "Denmark", "Djibouti", "Dominican Republic", "Ecuador", "Egypt",
            "El Salvador", "Estonia", "Ethiopia", "Fiji", "Finland", "France",
            "Georgia", "Germany", "Ghana", "Greece", "Guatemala", "Honduras",
            "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
            "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
            "Kazakhstan", "Kenya", "Kuwait", "Lebanon", "Lesotho", "Liberia",
            "Libya", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
            "Malaysia", "Maldives", "Malta", "Mauritius", "Mexico", "Mongolia",
            "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal",
            "Netherlands", "New Zealand", "Nigeria", "North Korea", "Norway",
            "Oman", "Pakistan", "Panama", "Papua New Guinea", "Paraguay",
            "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
            "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia",
            "Seychelles", "Singapore", "Slovakia", "Slovenia", "Somalia",
            "South Africa", "South Korea", "Spain", "Sri Lanka", "Sudan",
            "Sweden", "Switzerland", "Syria", "Taiwan", "Tanzania", "Thailand",
            "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
            "United Kingdom", "United States", "Uruguay", "Uzbekistan",
            "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
        ]

        # Match complete country names, longest names first.
        country_pattern = re.compile(
            r"(?<![A-Za-z])("
            + "|".join(
                re.escape(name)
                for name in sorted(country_names, key=len, reverse=True)
            )
            + r")(?![A-Za-z])",
            re.IGNORECASE,
        )

        matches = list(country_pattern.finditer(joined))
        records = []

        for i, match in enumerate(matches):
            country = match.group(1)
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
            details = joined[match.end():end_pos].strip(" ,:\n")
            if details:
                records.append(f"{country} {details}")

        if records:
            return records

    # Keep the existing general-document chunking behavior.
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
    """Search user-specific documents and rerank by keyword overlap."""
    import re

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

    if not (index.ntotal == len(chunks) == len(metadata)):
        raise ValueError("FAISS index, chunks and metadata counts do not match.")

    if index.ntotal == 0:
        return []

    embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(embedding, index.ntotal)
    stop_words = {
        "what", "is", "the", "a", "an", "of", "in", "to", "which",
        "used", "with", "for", "and", "how", "does", "do", "this",
    }
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - stop_words
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
        normalized = " ".join(chunk_text.split()).casefold()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        terms = set(re.findall(r"[a-z0-9]+", chunk_text.lower()))
        overlap = len(query_terms & terms) / max(1, len(query_terms))

        # Strongly prioritize exact technical terms when asked about them.
        technical_terms = {"faiss", "ollama", "fastapi", "python", "llama", "rag"}
        query_technical_terms = query_terms & technical_terms
        exact_technical_match = bool(
            query_technical_terms
            and any(
                re.search(rf"\b{re.escape(term)}\b", chunk_text, re.IGNORECASE)
                for term in query_technical_terms
            )
        )

        results.append({
            "text": chunk_text,
            "distance": distance,
            "filename": meta["filename"],
            "document_id": meta.get("document_id"),
            "user_id": meta.get("user_id"),
            "chunk_index": int(meta["chunk_index"]),
            "keyword_overlap": overlap,
            "exact_technical_match": exact_technical_match,
        })

    # Prioritize exact technical terms requested by the user.
    technical_terms = {"faiss", "ollama", "fastapi", "python", "llama", "rag"}
    query_technical_terms = query_terms & technical_terms

    if query_technical_terms:
        results.sort(
            key=lambda item: (
                -sum(
                    1 for term in query_technical_terms
                    if re.search(
                        rf"\b{re.escape(term)}\b",
                        item["text"],
                        re.IGNORECASE,
                    )
                ),
                item["distance"],
            )
        )
    else:
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
