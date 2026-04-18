# Task 01 — Project Setup (Next.js + FastAPI)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T01 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Setup project Next.js (frontend) dan FastAPI (backend API untuk ML inference) dalam monorepo structure.

## Acceptance Criteria
- [ ] Next.js app initialized dengan App Router (v14+)
- [ ] Tailwind CSS + shadcn/ui configured
- [ ] FastAPI backend initialized di `backend/` folder
- [ ] CORS configured antara frontend ↔ backend
- [ ] Both apps bisa running bersamaan (dev mode)
- [ ] Framer Motion installed

## Project Structure
```
skin_cancer_v3/
├── web/                          # Next.js Frontend
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx              # Landing page
│   │   ├── detect/
│   │   │   └── page.tsx          # Upload & Detection page
│   │   ├── results/
│   │   │   └── page.tsx          # Results display
│   │   ├── diseases/
│   │   │   └── [slug]/page.tsx   # Disease info (dynamic)
│   │   └── dashboard/
│   │       └── page.tsx          # Model comparison
│   ├── components/
│   │   ├── ui/                   # shadcn components
│   │   ├── ImageUploader.tsx
│   │   ├── PredictionCard.tsx
│   │   ├── GradCamViewer.tsx
│   │   ├── ConfidenceChart.tsx
│   │   ├── Navbar.tsx
│   │   ├── Footer.tsx
│   │   └── ThemeToggle.tsx
│   ├── lib/
│   │   ├── api.ts                # API client
│   │   └── utils.ts
│   ├── styles/
│   │   └── globals.css
│   ├── public/
│   ├── tailwind.config.ts
│   ├── next.config.js
│   └── package.json
│
├── backend/                      # FastAPI Backend
│   ├── main.py                   # FastAPI app
│   ├── routers/
│   │   └── predict.py            # Prediction endpoint
│   ├── services/
│   │   ├── predictor.py          # ML inference service
│   │   └── gradcam.py            # Grad-CAM service
│   ├── schemas/
│   │   └── prediction.py         # Pydantic models
│   └── requirements.txt
│
├── models/                       # Exported ML models
├── data/                         # Dataset
├── notebooks/                    # Jupyter notebooks
├── src/                          # ML source code
└── sprint_plans/                 # Sprint documentation
```

## Setup Commands

### Frontend (Next.js)
```bash
npx -y create-next-app@latest web --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --use-npm
cd web
npx -y shadcn@latest init
npm install framer-motion recharts lucide-react
npm install next-themes  # Dark/Light mode
```

### Backend (FastAPI)
```bash
mkdir backend
cd backend
pip install fastapi uvicorn python-multipart Pillow tensorflow numpy
# Create main.py with CORS middleware
```

### Run Development
```bash
# Terminal 1: Frontend
cd web && npm run dev  # http://localhost:3000

# Terminal 2: Backend
cd backend && uvicorn main:app --reload --port 8000  # http://localhost:8000
```

## Estimated Time
~2 jam

## Dependencies
- Node.js 18+ dan npm installed
- Python 3.9+ dan pip installed
