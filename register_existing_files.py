
from pathlib import Path

from app.database import SessionLocal
from app.models.document import Document
from app.rag.document_loader import extract_text_from_file
from app.rag.vector_store import rebuild_vector_store

UPLOAD_DIR = Path("app/rag/uploads")
USER_ID = 2

db = SessionLocal()

try:
    added = 0

    for file_path in sorted(UPLOAD_DIR.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {
            ".pdf", ".docx", ".txt", ".csv",
            ".xls", ".xlsx", ".jpg", ".jpeg", ".png"
        }:
            continue

        # Skip files that cannot produce usable text.
        try:
            text = extract_text_from_file(str(file_path))
        except Exception as exc:
            print(f"SKIPPED {file_path.name}: {exc}")
            continue

        if not text or not text.strip():
            print(f"SKIPPED {file_path.name}: no text extracted")
            continue

        # Avoid adding the same physical file twice.
        existing = (
            db.query(Document)
            .filter(Document.file_path == str(file_path))
            .first()
        )

        if existing:
            print(f"ALREADY REGISTERED: {file_path.name}")
            continue

        document = Document(
            user_id=USER_ID,
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            file_path=str(file_path),
        )

        db.add(document)
        added += 1

    db.commit()
    print(f"\nNew documents registered: {added}")

    documents = db.query(Document).all()
    result = rebuild_vector_store(documents)
    print("FAISS rebuild result:", result)

except Exception:
    db.rollback()
    raise

finally:
    db.close()
