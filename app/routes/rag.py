from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from ollama import chat

from app.rag.pdf_loader import extract_text_from_pdf
from app.rag.vector_store import (
    create_vector_store,
    search_vector_store,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)


# =========================================================
# UPLOAD DIRECTORY
# =========================================================

UPLOAD_DIR = Path("app/rag/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# PDF UPLOAD
# =========================================================

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        # Read uploaded file
        content = await file.read()

        # Save PDF
        with open(file_path, "wb") as f:
            f.write(content)

        # Extract text from PDF
        text = extract_text_from_pdf(str(file_path))

        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )

        # Create vector store
        create_vector_store(text)

        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "characters": len(text),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )


# =========================================================
# RAG SEARCH
# =========================================================

@router.get("/search")
def search(query: str):

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
            detail=f"Search failed: {str(e)}"
        )


# =========================================================
# RAG ASK - OLLAMA + LLAMA 3.2
# =========================================================

@router.get("/ask")
def ask(query: str):

    # -----------------------------------------------------
    # Validate query
    # -----------------------------------------------------

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # -----------------------------------------------------
    # Search relevant documents
    # -----------------------------------------------------

    try:
        results = search_vector_store(query)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )

    # -----------------------------------------------------
    # No results
    # -----------------------------------------------------

    if not results:
        return {
            "query": query,
            "answer": "I could not find relevant information in the uploaded document.",
            "sources": []
        }

    # -----------------------------------------------------
    # Build context from search results
    # -----------------------------------------------------

    context_parts = []

    for result in results:

        if isinstance(result, dict):

            # Try common text keys
            text = (
                result.get("text")
                or result.get("content")
                or result.get("page_content")
                or ""
            )

            if text:
                context_parts.append(str(text))

        elif isinstance(result, str):

            context_parts.append(result)

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # No usable text
    # -----------------------------------------------------

    if not context.strip():
        return {
            "query": query,
            "answer": "Relevant documents were found, but no text was available.",
            "sources": results
        }

    # -----------------------------------------------------
    # Send context + question to Ollama
    # -----------------------------------------------------

    try:

        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Answer the user's question using only the "
                        "information provided in the context. "
                        "Do not invent information. "
                        "If the answer is not available in the context, "
                        "say: "
                        "'I could not find this information in the uploaded document.'"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n\n"
                        f"{context}\n\n"
                        f"Question:\n"
                        f"{query}"
                    )
                }
            ]
        )

        answer = response.message.content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama AI response failed: {str(e)}"
        )

    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    return {
        "query": query,
        "answer": answer,
        "sources": results
    }
