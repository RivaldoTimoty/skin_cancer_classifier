from pathlib import Path
from src.loader import load_multiple
from src.chunking import chunk_documents
from src.retriever import build_index, save_index, load_index, get_context
from src.llm import generate, generate_stream, is_ollama_running

class RAGPipeline:
    def __init__(self):
        self.db = load_index()

    def process_files(self, file_paths: list[str], progress_callback=None) -> bool:
        """Loads, chunks, embeds, and saves index for multiple files."""
        try:
            raw_docs = load_multiple(file_paths)
            if not raw_docs:
                return False
            chunks = chunk_documents(raw_docs)
            self.db = build_index(chunks, progress_callback=progress_callback)
            save_index(self.db)
            return True
        except Exception as e:
            print(f"Error processing files: {e}")
            return False
            
    def has_index(self) -> bool:
        return self.db is not None

    def _build_prompt(self, context: str, question: str) -> str:
        return f"""Kamu adalah analis keuangan senior yang teliti.
Gunakan HANYA informasi dari konteks laporan keuangan berikut untuk menjawab pertanyaan.
Jika informasi tidak ada di konteks, jawab secara profesional bahwa kamu tidak menemuinya di dokumen.
Sertakan referensi (Sumber dan Halaman) jika memungkinkan.

Konteks:
{context}

Pertanyaan: {question}

Jawaban terstruktur:"""

    def query(self, question: str, model: str = "qwen2.5:7b", stream: bool = False):
        if stream:
            return self._query_stream(question, model)
        return self._query_sync(question, model)
        
    def _query_sync(self, question: str, model: str):
        if not self.db:
            return "Silakan upload dan proses dokumen terlebih dahulu."
        if not is_ollama_running():
            return "Error: Ollama tidak berjalan. Pastikan Ollama aktif."

        context = get_context(self.db, question, k=5)
        prompt = self._build_prompt(context, question)
        return generate(prompt, model=model)

    def _query_stream(self, question: str, model: str):
        if not self.db:
            yield "Silakan upload dan proses dokumen terlebih dahulu."
            return
        if not is_ollama_running():
            yield "Error: Ollama tidak berjalan. Pastikan Ollama aktif."
            return

        context = get_context(self.db, question, k=5)
        prompt = self._build_prompt(context, question)
        yield from generate_stream(prompt, model=model)

    def summarize(self, model: str = "qwen2.5:7b", stream: bool = False):
        question = "Buat ringkasan eksekutif (executive summary) dari dokumen ini, sebutkan poin-poin utamanya."
        return self.query(question, model, stream)

    def extract_kpi(self, model: str = "qwen2.5:7b", stream: bool = False):
        question = "Ekstrak Key Performance Indicators (KPI) ke dalam format bullet point/list, misalnya Pendapatan, Laba bersih, aset, hutang, margin, dll (jika ada informasinya)."
        return self.query(question, model, stream)

    def identify_risks(self, model: str = "qwen2.5:7b", stream: bool = False):
        question = "Identifikasi risiko utama (operasional, finansial, pasar) atau tantangan yang disebutkan dalam dokumen yang tercermin memengaruhi kinerja."
        return self.query(question, model, stream)
