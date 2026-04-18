# Task 04 — Setup Folder Structure

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 0 |
| **Task ID** | S0-T04 |
| **Priority** | High |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membuat struktur folder yang terorganisir mengikuti best practice ML project.

## Acceptance Criteria
- [ ] Semua folder berhasil dibuat
- [ ] Setiap folder berisi `__init__.py` (untuk Python packages) atau `.gitkeep`
- [ ] Struktur mengikuti standar ML project

## Folder Structure

```
skin_cancer_v3/
├── data/                    # Sudah ada
│   ├── Train/               # 9 class folders
│   └── Test/                # 9 class folders
├── notebooks/               # Jupyter notebooks untuk EDA & eksperimen
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predictor.py
│   └── gradcam.py
├── models/                  # Saved model files
├── reports/                 # Reports & documentation
│   └── figures/             # Generated plots & figures
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── sprint_plans/            # Sudah ada
├── web/                     # Next.js Frontend (Sprint 4)
├── backend/                 # FastAPI Backend (Sprint 4)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

## Implementation Steps
```bash
mkdir notebooks
mkdir src
mkdir models
mkdir reports
mkdir reports/figures
mkdir configs
mkdir tests
touch src/__init__.py
touch tests/__init__.py
```

## Estimated Time
~10 menit

## Dependencies
- Tidak ada dependency sebelumnya
