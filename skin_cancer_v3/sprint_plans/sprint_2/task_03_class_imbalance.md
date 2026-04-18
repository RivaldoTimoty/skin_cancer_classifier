# Task 03 — Handle Class Imbalance

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T03 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Mengimplementasikan strategi penanganan class imbalance yang telah ditentukan di Sprint 1 (S1-T09).

## Acceptance Criteria
- [ ] Class weights dihitung menggunakan sklearn
- [ ] Class weights diintegrasikan ke `model.fit()`
- [ ] Oversampling kelas minoritas (jika dipilih) diimplementasi
- [ ] Distribusi setelah balancing diverifikasi

## Implementation — Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(zip(np.unique(labels), class_weights))

# Gunakan di model.fit()
model.fit(
    train_dataset,
    class_weight=class_weight_dict,
    ...
)
```

## Implementation — Oversampling (Alternative)

```python
# Duplicate images dari kelas minoritas
import shutil

target_count = max(class_counts.values())  # Match the largest class

for cls, count in class_counts.items():
    if count < target_count:
        cls_path = os.path.join('data/Train', cls)
        images = os.listdir(cls_path)
        n_copies = target_count - count
        for i in range(n_copies):
            src = os.path.join(cls_path, images[i % len(images)])
            dst = os.path.join(cls_path, f"aug_{i}_{images[i % len(images)]}")
            shutil.copy2(src, dst)
```

## Expected Class Weights

| Kelas | Count | Weight (approx) |
|-------|-------|-----------------|
| pigmented benign keratosis | 462 | 0.54 |
| melanoma | 438 | 0.57 |
| basal cell carcinoma | 376 | 0.66 |
| nevus | 357 | 0.70 |
| squamous cell carcinoma | 181 | 1.37 |
| vascular lesion | 139 | 1.79 |
| actinic keratosis | 114 | 2.18 |
| dermatofibroma | 95 | 2.62 |
| seborrheic keratosis | 77 | 3.23 |

## Estimated Time
~1.5 jam

## Dependencies
- S1-T09 (strategi sudah ditentukan)
- S2-T01 (data pipeline)
