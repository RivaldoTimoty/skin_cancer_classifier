# Skin Cancer Detection using Deep Learning

## Overview
Project ini bertujuan untuk membangun sistem klasifikasi kanker kulit (9 kelas) dengan akurasi klinis tinggi. Menggunakan arsitektur Convolutional Neural Networks (CNN) dan Vision Transformers state-of-the-art (seperti EfficientNetV2, ConvNeXt, dan Swin Transformer), serta dilengkapi dengan **Next.js Frontend** dan **FastAPI Backend** yang profesional.

## Dataset
- **Sumber:** ISIC Archive (Modified 9-class dataset)
- **Classes:** 
  1. Actinic keratosis
  2. Basal cell carcinoma
  3. Dermatofibroma
  4. Melanoma
  5. Nevus
  6. Pigmented benign keratosis
  7. Seborrheic keratosis
  8. Squamous cell carcinoma
  9. Vascular lesion
- **Data Balance:** Highly Imbalanced.

## Tech Stack
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Modern Architectures:** `keras-cv`, `tensorflow-hub`
- **Backend API:** FastAPI (Python)
- **Frontend App:** Next.js + Tailwind CSS + shadcn/ui
- **Interpretability:** Grad-CAM (Gradient-weighted Class Activation Mapping)

## Quick Start
```bash
# 1. Activate Environment
# Windows
c:\Users\Asus\OneDrive\Documents\GitHub\Project\.venv\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Start Frontend (Next.js)
cd web
npm install
npm run dev

# 4. Start Backend (FastAPI)
cd backend
uvicorn main:app --reload
```

## Project Structure
```text
skin_cancer_v3/
├── backend/                 # FastAPI Backend 
├── web/                     # Next.js Frontend 
├── data/                    # Images (Train & Test)
├── notebooks/               # Jupyter notebooks (EDA)
├── src/                     # ML modules (data_loader, models, train)
├── models/                  # Exported models (.h5, .onnx)
├── sprint_plans/            # Project execution documentation
├── README.md                # Project read me
├── requirements.txt         # Dependencies
└── .gitignore               # Git rules
```

## Implementasi
- Project ini dibagi menjadi 5 fase Sprint (0 - 4). Lihat [Sprint Plans](sprint_plans/README.md) untuk detail perencanaan menyeluruh.
