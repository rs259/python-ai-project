
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import sqlite3

from ollama import chat
from app.routes.auth import verify_token

from app.rag.document_loader import extract_text_from_file
from app.rag.ocr import extract_text_from_image
from app.rag.vector_store import (
    create_vector_store,
    search_vector_store,
)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

# ============================================================
# CONFIGURATION
# ============================================================

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
async def upload_file(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required"
        )

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".csv",
        ".xls",
        ".xlsx", ".jpg", ".jpeg", ".png",
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported files: PDF, DOCX, TXT, CSV, XLS, XLSX, JPG, JPEG, PNG"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        text = extract_text_from_file(str(file_path))

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )

        create_vector_store(text, file.filename)

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_type": extension,
            "characters": len(text)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )


@router.get("/search")
async def search(query: str):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:

        results = search_vector_store(query)

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

        results = search_vector_store(query)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Step 2: Check results
    # --------------------------------------------------------

    if not results:

        raise HTTPException(
            status_code=404,
            detail="No relevant documents found"
        )

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
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not invent information.

Keep the answer clear and concise.

Context:
--------------------
{context}
--------------------

User Question:
{query}

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
