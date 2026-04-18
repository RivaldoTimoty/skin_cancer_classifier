# Task 06 — Color Channel Analysis (RGB Histogram)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T06 |
| **Priority** | Medium |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menganalisis distribusi masing-masing channel warna (Red, Green, Blue) untuk setiap kelas. Warna adalah fitur penting dalam dermatoskopi.

## Acceptance Criteria
- [ ] RGB histogram per kelas divisualisasikan
- [ ] Mean RGB values per kelas dihitung dan ditabelkan
- [ ] Apakah ada channel warna yang dominan per kelas diidentifikasi
- [ ] Rekomendasi color space (RGB vs HSV) ditentukan

## Implementation Steps
1. Sample gambar dari setiap kelas
2. Split per channel (R, G, B)
3. Plot overlapping histogram per channel per kelas
4. Hitung mean R, G, B per kelas
5. Buat tabel perbandingan

## Code Skeleton
```python
def plot_rgb_histogram(class_path, class_name, n_samples=30):
    r_vals, g_vals, b_vals = [], [], []
    images = os.listdir(class_path)[:n_samples]
    
    for img_name in images:
        img = cv2.imread(os.path.join(class_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        r_vals.extend(img[:,:,0].flatten())
        g_vals.extend(img[:,:,1].flatten())
        b_vals.extend(img[:,:,2].flatten())
    
    plt.hist(r_vals, bins=50, alpha=0.5, color='red', label='R')
    plt.hist(g_vals, bins=50, alpha=0.5, color='green', label='G')
    plt.hist(b_vals, bins=50, alpha=0.5, color='blue', label='B')
    plt.title(class_name)
    plt.legend()
```

## Expected Insights
- Melanoma biasanya lebih gelap → dominan nilai rendah di semua channel
- Vascular lesion → kemungkinan dominan di Red channel
- Informasi ini bisa membantu feature engineering

## Estimated Time
~30 menit

## Dependencies
- Sprint 0 selesai
