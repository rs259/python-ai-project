from fastapi import APIRouter, Depends, HTTPException
import sqlite3

from app.routes.auth import verify_token

router = APIRouter(
    prefix="/history",
    tags=["Chat History"]
)

DATABASE = "app.db"


# ============================================================
# GET CURRENT USER CHAT HISTORY
# ============================================================

@router.get("/")
def get_chat_history(
    user_id: int = Depends(verify_token)
):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                query AS question,
                answer,
                created_at
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

        conn.close()

        history = [dict(row) for row in rows]

        return {
            "count": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chat history: {str(e)}"
        )


# ============================================================
# GET CHAT HISTORY BY ID
# ============================================================

@router.get("/{history_id}")
def get_chat_by_id(
    history_id: int,
    user_id: int = Depends(verify_token)
):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                query AS question,
                answer,
                created_at
            FROM chat_history
            WHERE id = ?
            AND user_id = ?
            """,
            (history_id, user_id)
        )

        row = cursor.fetchone()

        conn.close()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Chat history not found"
            )

        return dict(row)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chat history: {str(e)}"
        )
@router.delete("/{history_id}")
def delete_chat_history(
    history_id: int,
    user_id: int = Depends(verify_token)
):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM chat_history
        WHERE id = ? AND user_id = ?
        """,
        (history_id, user_id)
    )

    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="Chat history not found"
        )

    conn.commit()
    conn.close()

    return {
        "message": "Chat history deleted successfully",
        "id": history_id
    }
