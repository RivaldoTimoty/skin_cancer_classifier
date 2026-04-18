# Sprint 4 — Next.js Web Application & Deployment

## Sprint Goal
*Membangun web application profesional dengan Next.js (frontend) + FastAPI (backend/ML serving), desain modern, premium, dan responsive.*

## Sprint Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Durasi** | 5-7 hari |
| **Total Story Points** | 42 SP |
| **Start Date** | TBD |
| **End Date** | TBD |
| **Status** | Belum Mulai |

## Tech Stack

| Layer | Technology | Alasan |
|-------|-----------|--------|
| **Frontend** | Next.js 14+ (App Router) | SSR, modern React, SEO-ready |
| **Styling** | Tailwind CSS + Framer Motion | Premium design system, animations |
| **UI Components** | shadcn/ui | Accessible, customizable components |
| **Backend API** | FastAPI (Python) | Serve ML model, image processing |
| **ML Inference** | TensorFlow / Keras | Load & predict with exported model |
| **State Management** | React hooks / Zustand | Lightweight state |
| **Charts** | Recharts / Chart.js | Prediction confidence visualization |
| **Deployment** | Vercel (FE) + Railway (BE) | Optimized for each |

## Design Direction

| Aspect | Specification |
|--------|--------------|
| **Theme** | Dark mode primary, light mode toggle |
| **Style** | Glassmorphism + gradient accents |
| **Colors** | Medical blue (#3B82F6) + teal (#14B8A6) + dark (#0F172A) |
| **Typography** | Inter / Plus Jakarta Sans (Google Fonts) |
| **Animations** | Framer Motion — page transitions, hover effects, loading states |
| **Layout** | Full-width hero, split-pane results, card grid for classes |
| **Mobile** | Mobile-first responsive design |

---

## Task List

| # | Task | Priority | SP | Status |
|---|------|----------|-----|--------|
| 1 | [Project Setup (Next.js + FastAPI)](./task_01_project_setup.md) | High | 3 | |
| 2 | [FastAPI Backend — ML Serving API](./task_02_fastapi_backend.md) | High | 5 | |
| 3 | [Design System & Theme](./task_03_design_system.md) | High | 3 | |
| 4 | [Landing Page / Hero Section](./task_04_landing_page.md) | High | 4 | |
| 5 | [Image Upload Component](./task_05_image_upload.md) | High | 3 | |
| 6 | [Prediction Results Page](./task_06_prediction_results.md) | High | 5 | |
| 7 | [Grad-CAM Heatmap Display](./task_07_gradcam_display.md) | Medium | 3 | |
| 8 | [Disease Information Pages](./task_08_disease_info.md) | Medium | 4 | |
| 9 | [Model Comparison Dashboard](./task_09_model_dashboard.md) | Medium | 3 | |
| 10 | [Responsive Design & Animations](./task_10_responsive_animations.md) | High | 3 | |
| 11 | [Error Handling & Loading States](./task_11_error_handling.md) | High | 2 | |
| 12 | [Deployment (Vercel + Railway)](./task_12_deployment.md) | Low | 4 | |

---

## Definition of Done
- [ ] Next.js app modern, premium, responsive berjalan
- [ ] FastAPI backend melayani prediction dengan < 2s response
- [ ] User experience seamless: upload → analyze → results → info
- [ ] Grad-CAM heatmap ditampilkan
- [ ] Dark/Light mode toggle
- [ ] Page transitions smooth (Framer Motion)
- [ ] SEO meta tags proper
- [ ] Mobile responsive (iPhone, iPad, Desktop)

## Dependencies
- Sprint 3 selesai (best model exported)
