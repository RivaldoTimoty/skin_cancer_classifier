# Sprint Plans — Skin Cancer Detection Project

Dokumentasi Sprint Planning untuk project deteksi kanker kulit menggunakan Deep Learning.

## 📅 Sprint Overview

| Sprint | Nama | Durasi | Tasks | SP | Status |
|--------|------|--------|-------|----|--------|
| [Sprint 0](./sprint_0/) | Project Setup & Environment | 1 hari | 6 | 6 | Belum Mulai |
| [Sprint 1](./sprint_1/) | Exploratory Data Analysis | 2-3 hari | 10 | 22 | Belum Mulai |
| [Sprint 2](./sprint_2/) | Data Pipeline & Baseline Model | 3-4 hari | 10 | 22 | Belum Mulai |
| [Sprint 3](./sprint_3/) | Multi-Model Exploration & Optimization | 5-7 hari | 17 | 58 | Belum Mulai |
| [Sprint 4](./sprint_4/) | Next.js Web App & Deployment | 5-7 hari | 12 | 42 | Belum Mulai |

**Total: 55 Tasks | 150 Story Points | Estimasi: 4-5 minggu**

---

## Models Explored (Sprint 3)

| # | Model | Params | Key Strength |
|---|-------|--------|-------------|
| 0 | Baseline CNN | ~2M | Reference benchmark |
| 1 | VGG16 | 138M | Simple, interpretable |
| 2 | ResNet50V2 | 25.6M | Skip connections, deep |
| 3 | InceptionV3 | 23.9M | Multi-scale features |
| 4 | DenseNet121 | 8M | Feature reuse, medical imaging champion |
| 5 | MobileNetV2 | 3.4M | Lightweight, mobile-ready |
| 6 | EfficientNetB3 | 12M | SOTA efficiency, ISIC winner |
| 7 | Xception | 22.9M | Texture recognition |

## Tech Stack (Sprint 4)

| Layer | Technology |
|-------|-----------|
| Frontend | **Next.js 14+** (App Router, TypeScript) |
| Styling | **Tailwind CSS** + **shadcn/ui** + **Framer Motion** |
| Backend | **FastAPI** (Python) |
| ML | **TensorFlow / Keras** |
| Charts | **Recharts** |
| Deploy | **Vercel** (FE) + **Railway** (BE) |

---

## Folder Structure

```
sprint_plans/
├── README.md                    ← File ini
├── sprint_0/ (7 files)         ← Setup & Environment
│   ├── overview.md
│   └── task_01 ~ task_06.md
├── sprint_1/ (11 files)        ← EDA
│   ├── overview.md
│   └── task_01 ~ task_10.md
├── sprint_2/ (11 files)        ← Baseline Model
│   ├── overview.md
│   └── task_01 ~ task_10.md
├── sprint_3/ (18 files)        ← Multi-Model Exploration
│   ├── overview.md
│   ├── task_01_model_documentation.md  (comprehensive!)
│   ├── task_02_vgg16.md
│   ├── task_03_resnet50v2.md
│   ├── task_04_inceptionv3.md
│   ├── task_05_densenet121.md
│   ├── task_06_mobilenetv2.md
│   ├── task_07_efficientnetb3.md
│   ├── task_08_xception.md
│   ├── task_09 ~ task_17.md
└── sprint_4/ (13 files)        ← Next.js Web App
    ├── overview.md
    └── task_01 ~ task_12.md
```
