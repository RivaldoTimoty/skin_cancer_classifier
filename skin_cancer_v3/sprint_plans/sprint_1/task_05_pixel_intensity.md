# Task 05 — Pixel Intensity Distribution Analysis

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T05 |
| **Priority** | Medium |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menganalisis distribusi intensitas pixel di seluruh dataset dan per kelas untuk memahami karakteristik brightness dan contrast gambar.

## Acceptance Criteria
- [ ] Histogram intensitas pixel keseluruhan dataset dibuat
- [ ] Histogram per kelas dibuat (overlay atau subplot)
- [ ] Mean brightness per kelas dihitung
- [ ] Apakah normalisasi khusus diperlukan ditentukan

## Implementation Steps
1. Sample gambar dari setiap kelas
2. Flatten pixel values dan buat histogram
3. Hitung mean, std intensity per kelas
4. Bandingkan distribusi antar kelas

## Code Skeleton
```python
import numpy as np
import cv2

def get_pixel_histogram(class_path, n_samples=50):
    all_pixels = []
    images = os.listdir(class_path)[:n_samples]
    for img_name in images:
        img = cv2.imread(os.path.join(class_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        all_pixels.extend(img.flatten())
    return all_pixels

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for i, cls in enumerate(sorted(classes)):
    pixels = get_pixel_histogram(os.path.join('data/Train', cls))
    ax = axes[i // 3][i % 3]
    ax.hist(pixels, bins=50, alpha=0.7, color='steelblue')
    ax.set_title(cls, fontsize=10)
    ax.set_xlim(0, 255)
```

## Expected Insights
- Apakah ada kelas yang cenderung lebih gelap/terang?
- Apakah distribusi pixel bervariasi signifikan antar kelas?
- Apakah histogram equalization bisa membantu?

## Estimated Time
~30 menit

## Dependencies
- Sprint 0 selesai
