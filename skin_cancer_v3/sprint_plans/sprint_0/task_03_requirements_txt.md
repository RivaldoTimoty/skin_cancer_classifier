# Task 03 — Buat requirements.txt

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 0 |
| **Task ID** | S0-T03 |
| **Priority** | High |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membuat file `requirements.txt` yang mendokumentasikan semua dependency dan versinya untuk reproducibility.

## Acceptance Criteria
- [ ] File `requirements.txt` dibuat di root project
- [ ] Semua dependency dengan versi spesifik tercatat
- [ ] Bisa diinstall ulang dengan `pip install -r requirements.txt` tanpa error

## Implementation Steps
1. Generate dari environment aktif:
   ```bash
   pip freeze > requirements.txt
   ```
2. **ATAU** buat manual dengan versi yang sudah ditentukan (lebih clean)
3. Test instalasi ulang:
   ```bash
   pip install -r requirements.txt
   ```

## 📄 Expected Output
```
tensorflow==2.15.0
opencv-python==4.9.0.80
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.2
plotly==5.18.0
pandas==2.2.0
numpy==1.26.3
Pillow==10.2.0
streamlit==1.30.0
albumentations==1.3.1
tensorboard==2.15.1
keras-cv==0.8.2
tensorflow-hub==0.16.1
```

## Estimated Time
~10 menit

## Dependencies
- S0-T02: Semua library sudah terinstall
