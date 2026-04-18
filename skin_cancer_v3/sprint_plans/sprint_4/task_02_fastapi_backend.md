# Task 02 — FastAPI Backend: ML Serving API

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T02 |
| **Priority** | High |
| **Story Points** | 5 |
| **Status** | Belum Mulai |

## Description
Membangun FastAPI backend yang melayani ML inference. Endpoints: predict (upload gambar → top predictions), gradcam (generate heatmap), dan disease info.

## Acceptance Criteria
- [ ] `POST /api/predict` — Upload gambar, return predictions + Grad-CAM
- [ ] `GET /api/diseases` — Return daftar 9 penyakit dengan detail
- [ ] `GET /api/models/info` — Return info model yang digunakan
- [ ] `GET /api/health` — Health check
- [ ] Model di-load sekali (singleton) → fast inference
- [ ] Response time < 2 detik
- [ ] CORS configured untuk Next.js frontend

## Implementation

```python
# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io, base64

app = FastAPI(title="Skin Cancer Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton model loader
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    from services.predictor import SkinCancerPredictor
    predictor = SkinCancerPredictor()

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    results = predictor.predict(image)
    gradcam_base64 = predictor.get_gradcam_base64(image)
    
    return {
        "predicted_class": results['predicted_class'],
        "confidence": results['confidence'],
        "top_predictions": [
            {"class": cls, "confidence": conf}
            for cls, conf in results['top_3']
        ],
        "all_probabilities": results['all_probabilities'],
        "gradcam_image": gradcam_base64,
        "risk_level": get_risk_level(results['predicted_class'])
    }

@app.get("/api/diseases")
async def get_diseases():
    return DISEASE_INFO  # Dictionary of all 9 diseases

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}
```

## API Response Schema

```json
{
  "predicted_class": "melanoma",
  "confidence": 0.87,
  "risk_level": "critical",
  "top_predictions": [
    {"class": "melanoma", "confidence": 0.87},
    {"class": "nevus", "confidence": 0.08},
    {"class": "pigmented benign keratosis", "confidence": 0.03}
  ],
  "all_probabilities": {
    "actinic keratosis": 0.01,
    "basal cell carcinoma": 0.005,
    "...": "..."
  },
  "gradcam_image": "data:image/png;base64,iVBOR..."
}
```

## Estimated Time
~4 jam

## Dependencies
- S3-T17 (exported model)
- S4-T01 (project setup)
