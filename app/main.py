from fastapi import FastAPI

from app.routes.users import router as users_router
from app.routes.auth import router as auth_router
from app.routes.rag import router as rag_router


app = FastAPI(
    title="Python AI Project",
    version="1.0.0",
    description="FastAPI + JWT Authentication + CRUD + RAG Project"
)


# =========================
# ROUTERS
# =========================

app.include_router(users_router)
app.include_router(auth_router)
app.include_router(rag_router)


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "Python AI Project is running",
        "version": "1.0.0",
        "features": [
            "JWT Authentication",
            "User CRUD",
            "PDF RAG"
        ]
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "python-ai-project"
    }
