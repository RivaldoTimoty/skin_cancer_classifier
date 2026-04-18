# Task 04 — Visualisasi Mean Image per Kelas

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T04 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menghitung dan menampilkan rata-rata gambar (mean image) untuk setiap kelas. Ini membantu memahami pola visual umum tiap kelas lesi kulit.

## Acceptance Criteria
- [ ] Mean image dihitung untuk setiap 9 kelas
- [ ] Grid visualization 3×3 menampilkan mean image semua kelas
- [ ] Perbedaan visual antar kelas didokumentasikan
- [ ] Plot disimpan ke `reports/figures/mean_images.png`

## Implementation Steps
1. Untuk setiap kelas:
   - Load semua gambar
   - Resize ke ukuran uniform (224×224)
   - Convert ke numpy array
   - Hitung mean pixel values
2. Tampilkan mean images dalam grid 3×3
3. Analisis pola visual

## Code Skeleton
```python
import numpy as np
import cv2
import os

TARGET_SIZE = (224, 224)

def compute_mean_image(class_path, target_size):
    images = []
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32))
    return np.mean(images, axis=0).astype(np.uint8)

# Compute and plot for all classes
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for i, cls in enumerate(sorted(classes)):
    mean_img = compute_mean_image(os.path.join('data/Train', cls), TARGET_SIZE)
    ax = axes[i // 3][i % 3]
    ax.imshow(mean_img)
    ax.set_title(cls, fontsize=10)
    ax.axis('off')
```

## Expected Insights
- Kelas dengan warna dominan gelap vs terang
- Distribusi spasial lesi (tengah vs pinggir)
- Kemiripan antar kelas yang mungkin menyulitkan klasifikasi

## Estimated Time
~45 menit

## Dependencies
- S1-T03 (dimensi gambar sudah dianalisis)
