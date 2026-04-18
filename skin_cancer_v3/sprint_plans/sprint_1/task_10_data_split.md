# Task 10 — Data Split (Train/Validation — Stratified)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T10 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membuat validation split dari training data menggunakan stratified sampling untuk memastikan proporsi kelas tetap seimbang di train dan validation set.

## Acceptance Criteria
- [ ] Training data di-split menjadi train (80%) dan validation (20%)
- [ ] Split dilakukan secara stratified (proporsi kelas sama)
- [ ] Distribusi kelas di train vs validation diverifikasi
- [ ] Random seed ditetapkan untuk reproducibility
- [ ] Path files disimpan untuk digunakan di pipeline

## Implementation Steps
1. Collect semua file paths dan labels
2. Gunakan `sklearn.model_selection.train_test_split` dengan `stratify`
3. Verifikasi distribusi
4. Simpan split sebagai CSV atau text file

## Code Skeleton
```python
from sklearn.model_selection import train_test_split
import os

# Collect all file paths and labels
file_paths = []
labels = []

for cls in os.listdir('data/Train'):
    cls_path = os.path.join('data/Train', cls)
    for img_name in os.listdir(cls_path):
        file_paths.append(os.path.join(cls_path, img_name))
        labels.append(cls)

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    file_paths, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Train: {len(X_train)} images")
print(f"Validation: {len(X_val)} images")

# Verify distribution
from collections import Counter
print("\nTrain distribution:")
print(Counter(y_train))
print("\nValidation distribution:")
print(Counter(y_val))
```

## Expected Output

| Set | Jumlah | Proporsi |
|-----|--------|----------|
| Train | ~1,791 | 80% |
| Validation | ~448 | 20% |
| Test | 118 | (existing) |

## Important Notes
- **JANGAN** shuffle test set atau mix train/test
- Random seed `42` digunakan untuk reproducibility
- Validation set digunakan untuk hyperparameter tuning
- Test set HANYA digunakan untuk evaluasi final

## Estimated Time
~30 menit

## Dependencies
- S1-T02 (distribusi kelas sudah diketahui)
