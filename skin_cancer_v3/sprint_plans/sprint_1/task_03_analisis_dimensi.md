# Task 03 — Analisis Dimensi & Resolusi Gambar

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T03 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menganalisis variasi dimensi (width, height) dan aspect ratio dari seluruh gambar dalam dataset untuk menentukan target resize yang optimal.

## Acceptance Criteria
- [ ] Distribusi width dan height divisualisasikan (histogram)
- [ ] Distribusi aspect ratio divisualisasikan
- [ ] Min, max, mean, median dimensi dicatat
- [ ] Target resize optimal ditentukan
- [ ] Scatter plot width vs height dibuat

## Implementation Steps
1. Iterate semua gambar di Train dan Test
2. Baca dimensi setiap gambar (tanpa load full image — gunakan `PIL.Image.open().size`)
3. Simpan data: filename, width, height, aspect_ratio, class
4. Buat visualisasi:
   - Histogram width & height
   - Scatter plot width vs height (color by class)
   - Box plot dimensi per kelas
5. Tentukan target resize berdasarkan analisis

## Code Skeleton
```python
from PIL import Image
import pandas as pd
import os

data = []
for cls in os.listdir('data/Train'):
    cls_path = os.path.join('data/Train', cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        with Image.open(img_path) as img:
            w, h = img.size
            data.append({
                'filename': img_name,
                'class': cls,
                'width': w,
                'height': h,
                'aspect_ratio': w / h
            })

df = pd.DataFrame(data)
print(df.describe())
```

## Expected Insights
- Apakah semua gambar ukuran sama atau bervariasi?
- Apakah ada outlier (gambar sangat besar/kecil)?
- Target resize yang direkomendasikan: **224×224** (standard) atau **300×300** (EfficientNet)

## Estimated Time
~45 menit

## Dependencies
- Sprint 0 selesai
