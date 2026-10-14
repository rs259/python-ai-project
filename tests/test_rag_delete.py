
import asyncio
from types import SimpleNamespace

import numpy as np


def test_delete_endpoint_exists():
    from app.routes.rag import router

    assert any(
        route.path == "/rag/documents/{document_id}"
        and "DELETE" in getattr(route, "methods", set())
        for route in router.routes
    )


def test_rebuild_excludes_deleted_document(tmp_path):
    from app.rag.vector_store import rebuild_vector_store

    uploads = tmp_path / "uploads"
    uploads.mkdir()

    file_one = uploads / "document_one.txt"
    file_two = uploads / "document_two.txt"

    file_one.write_text(
        "Python is a programming language.",
        encoding="utf-8",
    )
    file_two.write_text(
        "FastAPI is a Python web framework.",
        encoding="utf-8",
    )

    documents = [
        SimpleNamespace(
            id=101,
            user_id=1,
            filename="document_one.txt",
            file_path=str(file_one),
        ),
        SimpleNamespace(
            id=102,
            user_id=1,
            filename="document_two.txt",
            file_path=str(file_two),
        ),
    ]

    bundle_path = tmp_path / "vector_store.npz"

    result = rebuild_vector_store(
        documents,
        bundle_path=str(bundle_path),
    )

    assert result["total_documents"] == 2
    assert bundle_path.exists()

    # Simulate deletion from the database by rebuilding
    # with only the document that should remain.
    remaining_documents = [documents[1]]

    result = rebuild_vector_store(
        remaining_documents,
        bundle_path=str(bundle_path),
    )

    assert result["total_documents"] == 1

    with np.load(bundle_path, allow_pickle=True) as data:
        metadata = data["metadata"]
        document_ids = {
            int(item["document_id"])
            for item in metadata
        }

        assert document_ids == {102}
        assert len(data["chunks"]) == len(metadata)


def test_delete_endpoint_missing_document_does_not_delete_live_data(
    tmp_path, monkeypatch
):
    import app.routes.rag as rag

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.document import Document

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    monkeypatch.setattr(rag, "SessionLocal", TestSession)

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(rag, "UPLOAD_DIR", uploads)

    bundle_path = tmp_path / "vector_store.npz"
    np.savez(
        bundle_path,
        index_bytes=np.array([], dtype=np.uint8),
        chunks=np.array([], dtype=object),
        metadata=np.array([], dtype=object),
    )
    monkeypatch.setattr(rag, "VECTOR_BUNDLE_PATH", bundle_path)

    async def run_test():
        try:
            try:
                await rag.delete_document(
                    document_id=999999,
                    user_id=1,
                )
            except Exception as error:
                from fastapi import HTTPException

                assert isinstance(error, HTTPException)
                assert error.status_code == 404
            else:
                raise AssertionError("Expected document-not-found error")
        finally:
            engine.dispose()

    asyncio.run(run_test())


def test_delete_rolls_back_when_rebuild_fails(tmp_path, monkeypatch):
    import asyncio
    import app.routes.rag as rag

    from fastapi import HTTPException
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.database import Base
    from app.models.document import Document

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    document_file = uploads / "sample.txt"
    document_file.write_text("Test document", encoding="utf-8")

    bundle_path = tmp_path / "vector_store.npz"
    original_chunks = np.array(["original chunk"], dtype=object)
    original_metadata = np.array(
        [{"document_id": 123, "user_id": 1}],
        dtype=object,
    )
    np.savez(
        bundle_path,
        index_bytes=np.array([1, 2, 3], dtype=np.uint8),
        chunks=original_chunks,
        metadata=original_metadata,
    )

    with TestSession() as db:
        db.add(
            Document(
                id=123,
                user_id=1,
                filename="sample.txt",
                file_type="txt",
                file_path=str(document_file),
            )
        )
        db.commit()

    monkeypatch.setattr(rag, "SessionLocal", TestSession)
    monkeypatch.setattr(rag, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(rag, "VECTOR_BUNDLE_PATH", bundle_path)

    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("Simulated rebuild failure")

    monkeypatch.setattr(rag, "rebuild_vector_store", fail_rebuild)

    async def run_test():
        try:
            try:
                await rag.delete_document(document_id=123, user_id=1)
            except HTTPException as error:
                assert error.status_code == 500
            else:
                raise AssertionError("Expected deletion to fail")

            with TestSession() as db:
                assert db.query(Document).filter(Document.id == 123).first() is not None

            with np.load(bundle_path, allow_pickle=True) as data:
                assert data["chunks"].tolist() == ["original chunk"]
                assert data["metadata"][0]["document_id"] == 123

            assert document_file.exists()
        finally:
            engine.dispose()

    asyncio.run(run_test())

def test_delete_success_updates_database_bundle_and_file(
    tmp_path, monkeypatch
):
    import asyncio
    import app.routes.rag as rag

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.document import Document

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    document_file = uploads / "sample.txt"
    document_file.write_text("Sample test document", encoding="utf-8")

    bundle_path = tmp_path / "vector_store.npz"
    np.savez(
        bundle_path,
        index_bytes=np.array([1, 2, 3], dtype=np.uint8),
        chunks=np.array(["old chunk"], dtype=object),
        metadata=np.array(
            [{"document_id": 123, "user_id": 1}],
            dtype=object,
        ),
    )

    with TestSession() as db:
        db.add(
            Document(
                id=123,
                user_id=1,
                filename="sample.txt",
                file_type="txt",
                file_path=str(document_file),
            )
        )
        db.commit()

    monkeypatch.setattr(rag, "SessionLocal", TestSession)
    monkeypatch.setattr(rag, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(rag, "VECTOR_BUNDLE_PATH", bundle_path)

    def successful_rebuild(documents, bundle_path=None):
        assert list(documents) == []
        np.savez(
            bundle_path,
            index_bytes=np.array([], dtype=np.uint8),
            chunks=np.array([], dtype=object),
            metadata=np.array([], dtype=object),
        )

    monkeypatch.setattr(
        rag, "rebuild_vector_store", successful_rebuild
    )

    async def run_test():
        try:
            result = await rag.delete_document(
                document_id=123,
                user_id=1,
            )

            assert result["document_id"] == 123

            with TestSession() as db:
                assert (
                    db.query(Document)
                    .filter(Document.id == 123)
                    .first()
                    is None
                )

            with np.load(bundle_path, allow_pickle=True) as data:
                assert len(data["chunks"]) == 0
                assert len(data["metadata"]) == 0

            assert not document_file.exists()
        finally:
            engine.dispose()

    asyncio.run(run_test())

