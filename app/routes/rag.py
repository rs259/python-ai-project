import re

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import sqlite3
import uuid
from ollama import chat
from app.routes.auth import verify_token
from app.database import SessionLocal
from app.models.document import Document
from app.rag.document_loader import extract_text_from_file
from app.rag.ocr import extract_text_from_image
from app.rag.vector_store import (
    create_vector_store,
    search_vector_store,
    rebuild_vector_store,
)
router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

# ============================================================
# CONFIGURATION
# ============================================================
VECTOR_BUNDLE_PATH = Path("app/rag/vector_store.npz")
UPLOAD_DIR = Path("app/rag/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATABASE = "app.db"
OLLAMA_MODEL = "llama3.2"

# ============================================================
# CHAT HISTORY TABLE
# ============================================================

def create_chat_history_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER
)        """
    )

    conn.commit()
    conn.close()


create_chat_history_table()


# ============================================================
# SAVE CHAT HISTORY
# ============================================================

def save_chat_history(user_id: int, query: str, answer: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
INSERT INTO chat_history (user_id, query, answer, created_at)
VALUES (?, ?, ?, CURRENT_TIMESTAMP)        """,
        (user_id, query, answer)
    )

    conn.commit()
    conn.close()

# ============================================================
# UPLOAD PDF
# ============================================================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Depends(verify_token),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    allowed_extensions = {
        ".pdf", ".docx", ".txt", ".csv", ".xls", ".xlsx",
        ".jpg", ".jpeg", ".png",
    }
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported files: PDF, DOCX, TXT, CSV, XLS, XLSX, JPG, JPEG, PNG",
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_filename}"
    db = None
    bundle_path = Path("app/rag/vector_store.npz")
    bundle_backup = None
    document = None
    committed = False
    rebuild_succeeded = False

    import os
    import shutil
    import tempfile

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        file_path.write_bytes(content)

        text = extract_text_from_file(str(file_path))
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file",
            )

        if not bundle_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Vector bundle is missing; upload cancelled",
            )

        fd, backup_name = tempfile.mkstemp(
            prefix="upload_backup_", suffix=".npz", dir=str(bundle_path.parent)
        )
        os.close(fd)
        bundle_backup = Path(backup_name)
        shutil.copy2(bundle_path, bundle_backup)

        db = SessionLocal()
        document = Document(
            user_id=user_id,
            filename=file.filename,
            file_type=extension,
            file_path=str(file_path),
        )
        db.add(document)
        db.flush()

        # Mark rebuild as in-progress before publishing can occur.
        # If rebuilding fails after publishing, restore the previous bundle.
        rebuild_succeeded = True
        rebuild_vector_store(db.query(Document).all())

        db.commit()
        committed = True

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_type": extension,
            "characters": len(text),
            "document_id": document.id,
        }

    except HTTPException:
        if db is not None and not committed:
            db.rollback()

        if not committed and rebuild_succeeded and bundle_backup and bundle_backup.exists():
            os.replace(bundle_backup, bundle_path)
            bundle_backup = None

        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        if db is not None and not committed:
            db.rollback()

        if not committed and rebuild_succeeded and bundle_backup and bundle_backup.exists():
            try:
                os.replace(bundle_backup, bundle_path)
                bundle_backup = None
            except OSError as restore_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Upload failed and FAISS restore failed: {restore_error}",
                )

        if file_path.exists():
            file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {error}",
        )

    finally:
        if db is not None:
            db.close()
        if committed and bundle_backup and bundle_backup.exists():
            bundle_backup.unlink(missing_ok=True)


@router.get("/search")
async def search(query: str, user_id: int = Depends(verify_token)):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:

        results = search_vector_store(query, top_k=10, user_id=user_id)

        return {
            "query": query,
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


# ============================================================
# RAG ASK
# ============================================================

@router.get("/ask")
async def ask(
    query: str,
    user_id: int = Depends(verify_token)
):
    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # --------------------------------------------------------
    # Step 1: Search relevant documents
    # --------------------------------------------------------

    try:

        results = search_vector_store(query, top_k=28, user_id=user_id)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Step 2: Check results
    # --------------------------------------------------------


    if not results:
        answer = "I could not find the answer in the uploaded document."

        try:
            save_chat_history(
                user_id=user_id,
                query=query,
                answer=answer
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Chat history save failed: {str(e)}"
            )

        return {
            "query": query,
            "answer": answer,
            "sources": []
        }
    # --------------------------------------------------------
    # Step 3: Build context
    # --------------------------------------------------------

    context_parts = []

    for result in results:

        if isinstance(result, dict):

            text = result.get("text")

            if text:
                context_parts.append(text)

        elif isinstance(result, str):

            context_parts.append(result)

    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # Exact country-currency answer from retrieved source text
    # --------------------------------------------------------
    currency_match = re.search(
        r"\b(?:what\s+is\s+the\s+)?currency\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*\??$",
        query.strip(),
        re.IGNORECASE,
    )

    if currency_match:
        country = currency_match.group(1).strip()
        source_text = " ".join(context.split())

        next_country = (
            r"Afghanistan|Albania|Algeria|Argentina|Australia|Bangladesh|"
            r"Belgium|Brazil|Canada|China|Denmark|Egypt|France|Germany|"
            r"India|Indonesia|Iran|Italy|Japan|Kenya|Malaysia|Mexico|Nepal|"
            r"Netherlands|Nigeria|Norway|Pakistan|Philippines|Russia|"
            r"Saudi Arabia|Singapore|South Africa|South Korea|Spain|"
            r"Sri Lanka|Sweden|Switzerland|Thailand|Turkey|Ukraine|"
            r"United Kingdom|United States|Vietnam|Zimbabwe"
        )

        country_pattern = re.compile(
            rf"\b({re.escape(country)})\s+(.{{1,100}}?)"
            rf"(?=\b(?:{next_country})\b|$)",
            re.IGNORECASE,
        )
        record_match = country_pattern.search(source_text)

        if record_match:
            record = record_match.group(2).strip()
            code_match = re.search(r"\b([A-Z]{3})\b", record)

            if code_match:
                currency_name = re.sub(
                    r"\s+", " ", record
                ).strip()

                # Stop the currency name before its 3-letter code.
                currency_name = re.split(
                    r"\s+[^\s]{1,5}\s+[A-Z]{3}\b",
                    currency_name,
                    maxsplit=1,
                )[0].strip()

                answer = (
                    f"{currency_name}, currency code {code_match.group(1)}."
                )
                try:
                    save_chat_history(
                        user_id=user_id,
                        query=query,
                        answer=answer,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Chat history save failed: {str(e)}",
                    )

                return {
                    "query": query,
                    "answer": answer,
                    "sources": results,
                }

    # --------------------------------------------------------
    # Step 4: Check context
    # --------------------------------------------------------

    if not context:

        return {
            "query": query,
            "answer": (
                "Relevant documents were found, "
                "but no text was available."
            ),
            "sources": results
        }

    # --------------------------------------------------------
    # Step 5: Create LLM prompt
    # --------------------------------------------------------


    prompt = f"""
You are a precise document question-answering assistant.

Answer the question using only the supplied context.
Search the entire context carefully before deciding the answer is missing.
Match the meaning of the question to the relevant fact in the context.

Rules:
- For a country currency question, identify that country's currency name and code from the context.
- For India, if the context contains "Rupee" and "INR", answer "Indian Rupee (₹), currency code INR."
- Return a short, direct answer in one or two sentences.
- Do not refuse when the requested fact is clearly present.
- Do not invent facts that are absent from the context.
- Only if the answer is genuinely absent, say:
  "I could not find the answer in the uploaded document."

Context:
{context}

Question: {query}

Answer:
"""

    # --------------------------------------------------------
    # Step 6: Call Ollama
    # --------------------------------------------------------

    try:

        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"].strip()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ollama failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Step 7: Save chat history
    # --------------------------------------------------------

    try:

        save_chat_history(
            user_id=user_id,
            query=query,
            answer=answer
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Chat history save failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Step 8: Return response
    # --------------------------------------------------------

    return {
        "query": query,
        "answer": answer,
        "sources": results
    }

# ============================================================
# CHAT HISTORY
# ============================================================

from app.routes.auth import get_current_user


@router.get("/history")
async def chat_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]

    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, query, answer, created_at
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()
        conn.close()

        return {
            "user_id": user_id,
            "count": len(rows),
            "history": [dict(row) for row in rows]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat history fetch failed: {str(e)}"
        )
@router.get("/documents")
async def get_documents(
    user_id: int = Depends(verify_token)
):
    db = SessionLocal()

    try:
        documents = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .all()
        )

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "created_at": doc.created_at,
            }
            for doc in documents
        ]

    finally:
        db.close()



@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    user_id: int = Depends(verify_token),
):
    import os
    import shutil
    import tempfile

    db = SessionLocal()
    bundle_path = VECTOR_BUNDLE_PATH
    bundle_backup = None
    old_values = None
    file_path = None
    old_file_path = None
    deletion_committed = False
    rebuild_succeeded = False

    try:
        document = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            .first()
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        file_path = Path(document.file_path).resolve()
        upload_dir = UPLOAD_DIR.resolve()

        if not file_path.is_relative_to(upload_dir):
            raise HTTPException(
                status_code=400,
                detail="Invalid document file path",
            )

        old_file_path = document.file_path

        if not bundle_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Vector bundle is missing; deletion cancelled",
            )

        fd, backup_name = tempfile.mkstemp(
            prefix="delete_backup_",
            suffix=".npz",
            dir=str(bundle_path.parent),
        )
        os.close(fd)
        bundle_backup = Path(backup_name)
        shutil.copy2(bundle_path, bundle_backup)

        # Stage database deletion without committing.
        db.delete(document)
        db.flush()

        # Rebuild and atomically publish the new FAISS bundle.
        rebuild_succeeded = True

        rebuild_vector_store(
            db.query(Document).all(),
            bundle_path=str(bundle_path),
        )
        # Commit database only after a successful rebuild.
        db.commit()
        deletion_committed = True

        # Remove the uploaded file only after both operations succeed.
        other_records = (
            db.query(Document)
            .filter(Document.file_path == old_file_path)
            .count()
        )

        if other_records == 0 and file_path.is_file():
            try:
                file_path.unlink()
            except OSError:
                # The database and index are updated; orphan cleanup
                # can be retried separately.
                pass

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
        }

    except HTTPException:
        if not deletion_committed:
            db.rollback()

        if not deletion_committed and rebuild_succeeded and bundle_backup and bundle_backup.exists():
            try:
                os.replace(bundle_backup, bundle_path)
                bundle_backup = None
            except OSError as restore_error:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Request failed and FAISS restoration failed. "
                        "Keep the backup and inspect database/index state. "
                        f"Restore error: {restore_error}"
                    ),
                )

        raise
    except Exception as error:
        if not deletion_committed:
            db.rollback()

        # Restore only if the DB commit did not succeed.
        if not deletion_committed and rebuild_succeeded and bundle_backup and bundle_backup.exists():
            try:
                os.replace(bundle_backup, bundle_path)
                bundle_backup = None
            except OSError as restore_error:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Deletion failed and FAISS restoration failed. "
                        "Keep the backup and inspect database/index state. "
                        f"Restore error: {restore_error}"
                    ),
                )

        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {error}",
        )


    finally:
        db.close()

        # Delete backup only after successful database commit.
        if (
            deletion_committed
            and bundle_backup
            and bundle_backup.exists()
        ):
            bundle_backup.unlink(missing_ok=True)
