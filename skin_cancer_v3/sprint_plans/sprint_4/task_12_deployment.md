# Task 12 — Deployment (Vercel + Railway)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T12 |
| **Priority** | Low |
| **Story Points** | 4 |
| **Status** | Belum Mulai |

## Description
Deploy Next.js frontend ke Vercel dan FastAPI backend ke Railway (atau Hugging Face Spaces).

## Acceptance Criteria
- [ ] Frontend deployed to Vercel (or similar)
- [ ] Backend deployed to Railway / HF Spaces
- [ ] API URL configured as environment variable
- [ ] Model file successfully uploaded to backend
- [ ] End-to-end test on production
- [ ] Custom domain (optional)
- [ ] SSL/HTTPS enabled

## Deployment Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│  Vercel          │  API    │  Railway / HF Spaces  │
│  (Next.js)       │───────→│  (FastAPI + Model)     │
│  Frontend        │  HTTPS │  Backend               │
│  Static + SSR    │        │  ML Inference          │
└─────────────────┘         └──────────────────────┘
```

## Deployment Steps

### Frontend (Vercel)
```bash
# 1. Push to GitHub
git add . && git commit -m "Deploy" && git push

# 2. Connect Vercel
# - Go to vercel.com
# - Import repository
# - Set Framework: Next.js
# - Set Root Directory: web/
# - Add Environment Variable:
#   NEXT_PUBLIC_API_URL=https://your-backend.railway.app

# 3. Deploy (automatic on push)
```

### Backend (Railway)
```bash
# backend/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ../models/best_model.h5 ./models/
COPY ../models/model_config.json ./models/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Alternative: Hugging Face Spaces
```bash
# Create a Gradio/FastAPI Space
# Upload model via Git LFS
git lfs install
git lfs track "*.h5"
git add . && git push
```

## Considerations
- Model file ~48-100MB → use Git LFS or cloud storage
- Railway free tier: 500 hours/month, 512MB RAM
- HF Spaces: Free GPU (T4) for Gradio apps
- Vercel: Free for hobby projects

## Estimated Time
~4 jam

## Dependencies
- All Sprint 4 tasks completed
