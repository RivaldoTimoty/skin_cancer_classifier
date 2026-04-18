# Task 01 — Visualisasi Sample Gambar per Kelas

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T01 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menampilkan sample gambar dari setiap kelas (9 kelas) untuk mendapatkan pemahaman visual tentang tampilan masing-masing jenis lesi kulit.

## Acceptance Criteria
- [ ] Grid plot menampilkan minimal 5 sample gambar per kelas
- [ ] Setiap gambar diberi label kelas
- [ ] Plot disimpan sebagai figure di `reports/figures/`
- [ ] Observasi visual dicatat dalam notebook

## Implementation Steps
1. Load 5 random gambar dari setiap kelas
2. Buat grid plot `9 x 5` (9 kelas × 5 sample)
3. Tambahkan judul per baris dengan nama kelas
4. Simpan plot sebagai `reports/figures/sample_images_per_class.png`
5. Catat observasi visual:
   - Apakah kelas mudah dibedakan secara visual?
   - Apakah ada kemiripan antar kelas?
   - Apakah ada noise atau artefak pada gambar?

## Code Skeleton
```python
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

classes = os.listdir('data/Train')
fig, axes = plt.subplots(9, 5, figsize=(15, 27))

for i, cls in enumerate(sorted(classes)):
    cls_path = os.path.join('data/Train', cls)
    images = random.sample(os.listdir(cls_path), 5)
    for j, img_name in enumerate(images):
        img = Image.open(os.path.join(cls_path, img_name))
        axes[i][j].imshow(img)
        axes[i][j].axis('off')
        if j == 0:
            axes[i][j].set_title(cls, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/sample_images_per_class.png', dpi=150)
plt.show()
```

## Estimated Time
~30 menit

## Dependencies
- Sprint 0 selesai
- Matplotlib, Pillow terinstall
