import os
import fitz
import pdfplumber
import pandas as pd
from pathlib import Path
from docx import Document


def load_pdf(file_path: str) -> list[dict]:
    docs = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                docs.append({
                    "content": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "page": i + 1,
                        "type": "pdf"
                    }
                })
        doc.close()
    except Exception:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": Path(file_path).name,
                            "page": i + 1,
                            "type": "pdf"
                        }
                    })
    return docs


def load_csv(file_path: str) -> list[dict]:
    df = pd.read_excel(file_path) if file_path.endswith((".xlsx", ".xls")) else pd.read_csv(file_path)
    docs = []
    col_info = f"Kolom: {', '.join(df.columns.tolist())}\nJumlah baris: {len(df)}\n\n"
    summary = col_info + df.describe(include="all").to_string()
    docs.append({
        "content": summary,
        "metadata": {"source": Path(file_path).name, "page": 1, "type": "table_summary"}
    })
    chunk_size = 50
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        docs.append({
            "content": chunk.to_string(index=False),
            "metadata": {
                "source": Path(file_path).name,
                "page": start // chunk_size + 2,
                "type": "table_chunk",
                "rows": f"{start}–{min(start + chunk_size, len(df))}"
            }
        })
    return docs


def load_docx(file_path: str) -> list[dict]:
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    return [{
        "content": full_text,
        "metadata": {"source": Path(file_path).name, "page": 1, "type": "docx"}
    }]


def load_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [{
        "content": text,
        "metadata": {"source": Path(file_path).name, "page": 1, "type": "txt"}
    }]


def load_file(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".csv": load_csv,
        ".xlsx": load_csv,
        ".xls": load_csv,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
    }
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"Format file tidak didukung: {ext}")
    return loader(file_path)


def load_multiple(file_paths: list[str]) -> list[dict]:
    all_docs = []
    for path in file_paths:
        docs = load_file(path)
        all_docs.extend(docs)
    return all_docs
