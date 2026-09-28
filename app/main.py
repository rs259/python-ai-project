from fastapi import FastAPI

app = FastAPI(title="Python AI Project")

@app.get("/")
def home():
    return {"message": "Python project is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
