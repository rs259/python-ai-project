from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.users import router as users_router
from app.routes.auth import router as auth_router
from app.routes.rag import router as rag_router
from app.routes.history import router as history_router


app = FastAPI(
    title="Python AI Project",
    version="1.0.0",
    description="FastAPI + JWT Authentication + CRUD + RAG Project"
)


# ================================
# API ROUTES
# ================================

app.include_router(users_router)
app.include_router(auth_router)
app.include_router(rag_router)
app.include_router(history_router)


# ================================
# STATIC FRONTEND
# ================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ================================
# FRONTEND HOME
# ================================

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse("static/index.html")


# ================================
# HEALTH CHECK
# ================================

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }
