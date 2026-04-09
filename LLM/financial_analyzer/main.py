import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from src.pipeline import RAGPipeline
from src.llm import is_ollama_running

app = FastAPI(title="Financial Analyzer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
UPLOAD_PROGRESS = 0

# Ensure directories exist
FRONTEND_DIR = Path("frontend")
FRONTEND_DIR.mkdir(exist_ok=True)
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    model: str = "qwen2.5:7b"

class PresetRequest(BaseModel):
    type: str  # "summary", "kpi", "risk"
    model: str = "qwen2.5:7b"

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return "Frontend not found. Please create frontend/index.html"
    return index_path.read_text(encoding="utf-8")

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok", 
        "ollama_running": is_ollama_running(),
        "has_index": pipeline.has_index()
    }

@app.get("/api/progress")
async def get_progress():
    global UPLOAD_PROGRESS
    return {"progress": UPLOAD_PROGRESS}

@app.get("/api/files")
async def get_files():
    files = []
    if RAW_DIR.exists():
        for f in RAW_DIR.iterdir():
            if f.is_file():
                files.append({"name": f.name})
    return files

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global UPLOAD_PROGRESS
    UPLOAD_PROGRESS = 0
    saved_paths = []
    
    for file in files:
        file_path = RAW_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(str(file_path))
        
    def progress_callback(p):
        global UPLOAD_PROGRESS
        UPLOAD_PROGRESS = p

    success = pipeline.process_files(saved_paths, progress_callback=progress_callback)
    UPLOAD_PROGRESS = 100
    
    if success:
        return {"message": f"Successfully processed {len(files)} files."}
    else:
        raise HTTPException(status_code=500, detail="Failed to process documents.")

@app.post("/api/chat")
async def chat(req: QueryRequest):
    if not pipeline.has_index():
        raise HTTPException(status_code=400, detail="Harap upload laporan keuangan terlebih dahulu.")
        
    async def event_generator():
        try:
            # We use the sync generator in a wrapper
            for chunk in pipeline.query(req.query, model=req.model, stream=True):
                yield chunk
                await asyncio.sleep(0.001)
        except Exception as e:
            yield f"\n\nError: {str(e)}"
            
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/api/preset")
async def preset(req: PresetRequest):
    if not pipeline.has_index():
        raise HTTPException(status_code=400, detail="Harap upload laporan keuangan terlebih dahulu.")
        
    async def event_generator():
        try:
            if req.type == "summary":
                generator = pipeline.summarize(model=req.model, stream=True)
            elif req.type == "kpi":
                generator = pipeline.extract_kpi(model=req.model, stream=True)
            elif req.type == "risk":
                generator = pipeline.identify_risks(model=req.model, stream=True)
            else:
                yield "Tipe preset tidak dikenali."
                return
                
            for chunk in generator:
                yield chunk
                await asyncio.sleep(0.001)
        except Exception as e:
            yield f"\n\nError: {str(e)}"
            
    return StreamingResponse(event_generator(), media_type="text/plain")
