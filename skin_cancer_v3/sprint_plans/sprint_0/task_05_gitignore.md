# Task 05 — Buat .gitignore

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 0 |
| **Task ID** | S0-T05 |
| **Priority** | Medium |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membuat file `.gitignore` yang comprehensive untuk menghindari file yang tidak perlu masuk ke repository.

## Acceptance Criteria
- [ ] File `.gitignore` dibuat di root project
- [ ] Virtual environment di-ignore
- [ ] Model files besar di-ignore (opsional, tergantung kebutuhan)
- [ ] File cache Python di-ignore
- [ ] Jupyter notebook checkpoints di-ignore
- [ ] File OS-specific di-ignore

## Content .gitignore

```gitignore
# Virtual Environment
venv/
env/
.env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
desktop.ini

# Model files (uncomment jika model terlalu besar)
# models/*.h5
# models/*.pb
# models/*.tflite

# Data (uncomment jika data tidak perlu di-track)
# data/

# TensorBoard logs
logs/
tensorboard_logs/

# Temporary files
*.tmp
*.bak
*.log

# Environment variables
.env
.env.local
```

## Estimated Time
~5 menit

## Dependencies
- Tidak ada dependency sebelumnya
