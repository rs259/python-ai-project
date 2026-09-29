from fastapi import FastAPI

from app.routes.users import router as users_router
from app.routes.auth import router as auth_router


app = FastAPI(
    title="Python AI Project",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Python AI Project is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/db-test")
def db_test():
    return {
        "database": "connected"
    }


app.include_router(users_router)
app.include_router(auth_router)
