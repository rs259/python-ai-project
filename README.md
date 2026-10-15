# Python AI RAG Project

A Python AI application built using FastAPI, FAISS, and Sentence Transformers.

## Features
- FastAPI REST API
- User authentication
- Document upload and management
- Text extraction and chunking
- Semantic search using FAISS
- SQLite database
- Automated tests

## Technologies
- Python
- FastAPI
- SQLAlchemy
- SQLite
- FAISS
- Sentence Transformers
- Ollama

## Installation

Clone the project:

```bash
git clone https://github.com/ravishekhar259-lgtm/test-project.git
cd test-project
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
uvicorn app.main:app --reload
```

Open API documentation:

http://127.0.0.1:8000/docs

## Run Tests

```bash
PYTHONPATH=. pytest -q
```

## Safety
Do not commit passwords, API keys, local databases, or private uploaded documents.

## Project Status
Under development.
