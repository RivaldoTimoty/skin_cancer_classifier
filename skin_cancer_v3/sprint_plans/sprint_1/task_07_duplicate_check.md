# Task 07 — Duplicate & Corrupt Image Check

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T07 |
| **Priority** | Medium |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Memeriksa apakah ada gambar yang duplikat atau corrupt dalam dataset. Gambar duplikat bisa menyebabkan data leakage, sedangkan gambar corrupt bisa menyebabkan error saat training.

## Acceptance Criteria
- [ ] Semua gambar dicek apakah bisa dibuka tanpa error
- [ ] Gambar corrupt (jika ada) diidentifikasi dan didokumentasikan
- [ ] Duplicate detection menggunakan image hashing
- [ ] Jumlah duplikat per kelas dicatat
- [ ] Rekomendasi tindakan untuk duplikat/corrupt

## Implementation Steps
1. **Corrupt Check:**
   - Coba buka setiap gambar dengan Pillow
   - Catat yang gagal dibuka
2. **Duplicate Check (Image Hashing):**
   - Resize semua gambar ke 8×8
   - Convert ke grayscale
   - Compute perceptual hash (pHash)
   - Cari hash yang identik
3. Dokumentasi temuan

## Code Skeleton
```python
from PIL import Image
import hashlib
import os

def check_corrupt_images(data_dir):
    corrupt = []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_path):
            try:
                img = Image.open(os.path.join(cls_path, img_name))
                img.verify()
            except Exception as e:
                corrupt.append((cls, img_name, str(e)))
    return corrupt

def compute_image_hash(img_path):
    img = Image.open(img_path).resize((8, 8)).convert('L')
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ''.join('1' if p > avg else '0' for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()

def find_duplicates(data_dir):
    hashes = {}
    duplicates = []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            h = compute_image_hash(img_path)
            if h in hashes:
                duplicates.append((img_name, hashes[h]))
            else:
                hashes[h] = (cls, img_name)
    return duplicates
```

## Estimated Time
~30 menit

## Dependencies
- Sprint 0 selesai

## Notes
- Jika banyak duplikat ditemukan, perlu strategi removal sebelum training
- Cross-set duplicates (train → test) = **data leakage** yang HARUS dihapus
