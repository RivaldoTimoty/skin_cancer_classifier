from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(raw_docs: list[dict], chunk_size: int = 600, chunk_overlap: int = 80) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    langchain_docs = [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in raw_docs
        if d["content"].strip()
    ]
    chunks = splitter.split_documents(langchain_docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
