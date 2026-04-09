import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embedding import get_embedding_model


VECTOR_STORE_PATH = Path(__file__).parent.parent / "vector_store"


def build_index(chunks: list[Document], progress_callback=None) -> FAISS:
    embeddings = get_embedding_model()
    batch_size = 32
    total = len(chunks)
    db = None
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
            
        if progress_callback:
            progress = min(100, int(((i + len(batch)) / total) * 100))
            progress_callback(progress)
            
    return db


def save_index(db: FAISS, name: str = "faiss_index") -> None:
    VECTOR_STORE_PATH.mkdir(exist_ok=True)
    db.save_local(str(VECTOR_STORE_PATH / name))


def load_index(name: str = "faiss_index") -> FAISS | None:
    index_path = VECTOR_STORE_PATH / name
    if not index_path.exists():
        return None
    embeddings = get_embedding_model()
    return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)


def retrieve(db: FAISS, query: str, k: int = 5) -> list[Document]:
    return db.similarity_search(query, k=k)


def get_context(db: FAISS, query: str, k: int = 5) -> str:
    docs = retrieve(db, query, k)
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Sumber: {source}, Halaman: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
